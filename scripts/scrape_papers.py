import logging
import sqlite3
from datetime import date, datetime
from itertools import batched
from pathlib import Path
from typing import Annotated, Any, Final, cast

import pandas as pd
import requests
from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, PlainValidator, model_validator
from requests import Session
from requests.adapters import HTTPAdapter, Retry
from tqdm.auto import tqdm

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DATA_FOLDER: Final = Path(__file__).parent.parent / "data"
PAPERS_METADATA_DB_PATH = DATA_FOLDER / "papers_metadata.db"


EXTRA_FIELDS: Final = [
    "title",
    "externalIds",
    "s2FieldsOfStudy",  # human supplied + model decided
    "authors.authorId",
    "citationCount",
    "referenceCount",
    "citations.paperId",
    "references.paperId",
    "publicationDate",
    "influentialCitationCount",
    "isOpenAccess",
    "openAccessPdf",  # url, status
    "tldr",  # model, text
]


def custom_id_deserializer(entry: list[str] | str | list[dict[str, str | None]]) -> list[str]:
    if isinstance(entry, list):
        if all(isinstance(idx, dict) for idx in entry):
            entry = cast(list[dict[str, str | None]], entry)
            ids: list[str] = [value for d in entry if (value := next(iter(d.values())))]
        elif all(isinstance(idx, str) for idx in entry):
            ids = cast(list[str], entry)
        else:
            raise ValueError(f"Mixed type of entry {entry}")
    elif isinstance(entry, str):
        ids = entry.split(",")
    else:
        raise ValueError(f"Unsupported type of entry {type(entry)}")
    return ids


SerializableStrList = Annotated[
    list[str],
    PlainSerializer(lambda x: ",".join(x)),
    PlainValidator(custom_id_deserializer),
]


class PaperMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    paper_id: str = Field(alias="paperId")
    title: str
    arxiv_id: str | None  # from externalIds
    fields_of_study: SerializableStrList = Field(alias="fieldsOfStudy")
    authors_ids: SerializableStrList = Field(alias="authors")
    citation_count: int = Field(alias="citationCount")
    reference_count: int = Field(alias="referenceCount")
    citations_ids: SerializableStrList = Field(alias="citations")
    references_ids: SerializableStrList = Field(alias="references")
    publication_date: date | None = Field(alias="publicationDate")
    influential_citation_count: int = Field(alias="influentialCitationCount")
    is_open_access: bool = Field(alias="isOpenAccess")
    open_access_pdf_url: str | None
    open_access_pdf_status: str | None
    tldr_text: str | None
    # timestamp when it has been retrieved from Semantic Scholar
    # so that we know when our local db gets outdated
    retrieval_time: datetime = Field(default_factory=datetime.now)

    @model_validator(mode="before")
    @classmethod
    def preprocess(cls, data: dict[str, Any]) -> dict[str, Any]:
        # extract s2FieldsOfStudy
        if "s2FieldsOfStudy" in data:
            fields_of_study: list[str] = sorted(list({field["category"] for field in data["s2FieldsOfStudy"]}))
            data.update({"fields_of_study": fields_of_study})

        # extract arxiv_id
        if "externalIds" in data:
            external_ids: dict[str, str] = data["externalIds"]
            arxiv_id = external_ids.get("ArXiv")
            data.update({"arxiv_id": arxiv_id})

        # flatten openAccessPdf
        if "openAccessPdf" in data:
            if data["openAccessPdf"]:
                open_access_pdf: dict[str, str] = data["openAccessPdf"]
                open_access_pdf_url = open_access_pdf["url"]
                open_access_pdf_status = open_access_pdf["status"]
            else:
                open_access_pdf_url = None
                open_access_pdf_status = None
            data.update(
                {
                    "open_access_pdf_url": open_access_pdf_url,
                    "open_access_pdf_status": open_access_pdf_status,
                }
            )

        # extract tldr text
        if "tldr" in data:
            tldr: dict[str, str | None] | None = data["tldr"]
            tldr_text = tldr.get("text") if tldr is not None else None
            data.update({"tldr_text": tldr_text})

        return data


# TODO: cleanup
def get_papers_metadata_from_semantic_scholar(papers_ids: list[str]) -> list[PaperMetadata]:
    # https://api.semanticscholar.org/api-docs/#tag/Paper-Data/operation/post_graph_get_papers
    batch_endpoint = "https://api.semanticscholar.org/graph/v1/paper/batch"
    with Session() as session:
        # Semantic Scholar asks to be considerate and use exponential backoff
        retries = Retry(
            total=15,
            backoff_factor=10.0,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        session.mount("http://", HTTPAdapter(max_retries=retries))
        session.mount("https://", HTTPAdapter(max_retries=retries))
        response = session.post(
            batch_endpoint,
            params={"fields": ",".join(EXTRA_FIELDS)},
            json={"ids": papers_ids},
            timeout=60,
        )
    if response.status_code == requests.codes.OK:
        data = response.json()
        papers_metadata = [PaperMetadata(**entry) for entry in data if entry is not None]
        return papers_metadata
    raise RuntimeError(f"Retrieval failed with code: {response.status_code}")


def get_citations_graph(origin_paper_id: str, banned_words: list[str], banned_paper_ids: set[str], depth_limit: int = 10) -> list[PaperMetadata]:
    def _fetch_from_local(_papers_ids: list[str]) -> tuple[list[PaperMetadata], list[str]]:
        local_paper_ids: set[str] = set(known_papers_metadata_df.paper_id.values)
        _to_retrieve_papers_ids = local_paper_ids.intersection(_papers_ids)
        _remaining_papers_ids = list(set(_papers_ids) - _to_retrieve_papers_ids)

        _papers_metadata_df: pd.DataFrame = known_papers_metadata_df.loc[
            known_papers_metadata_df.paper_id.isin(_to_retrieve_papers_ids)
        ]
        _papers_metadata_df = _papers_metadata_df.loc[~_papers_metadata_df.title.str.contains(banned_pattern, case=False)]
        _papers_metadata = [
            PaperMetadata(**cast(dict[str, Any], record))
            for record in _papers_metadata_df.to_dict(orient="records")
        ]

        return _papers_metadata, _remaining_papers_ids

    def _fetch_from_semantic_scholar(_papers_ids: list[str]) -> list[PaperMetadata]:
        # TODO: it is preferable to have dynamic batch size, due to citation limits (9999)
        batch_size = 200
        _papers_metadata: list[PaperMetadata] = []
        for papers_ids in batched(tqdm(_papers_ids), n=batch_size):
            _papers_metadata_batch = get_papers_metadata_from_semantic_scholar(list(papers_ids))
            # TODO: rethink this detour
            _papers_metadata_df = pd.DataFrame([metadata.model_dump() for metadata in _papers_metadata_batch])
            _papers_metadata_df = _papers_metadata_df.loc[~_papers_metadata_df.title.str.contains(banned_pattern, case=False)]
            _papers_metadata_batch = [
                PaperMetadata(**cast(dict[str, Any], record))
                for record in _papers_metadata_df.to_dict(orient="records")
            ]
            _papers_metadata.extend(_papers_metadata_batch)
            save_papers_metadata_to_db(_papers_metadata_batch)
        return _papers_metadata

    known_papers_metadata_df = read_papers_metadata_from_db()
    banned_pattern = "|".join(banned_words)  # OR for regex

    papers_metadata: list[PaperMetadata] = []
    processed_papers_ids: set[str] = set()
    papers_ids = [origin_paper_id]
    depth = 0
    while papers_ids:
        logger.info(f"Processing {len(papers_ids)} papers at depth {depth}")
        papers_metadata_from_local, remaining_papers_ids = _fetch_from_local(papers_ids)
        papers_metadata_from_semantic_scholar = _fetch_from_semantic_scholar(remaining_papers_ids)
        papers_metadata_at_depth = (
            papers_metadata_from_local + papers_metadata_from_semantic_scholar
        )
        processed_papers_ids.update(
            paper_metadata.paper_id for paper_metadata in papers_metadata_at_depth
        )
        papers_metadata.extend(papers_metadata_at_depth)
        candidate_papers_ids = {
            citation_id
            for paper_metadata in papers_metadata_at_depth
            for citation_id in paper_metadata.citations_ids
            if citation_id
        }

        papers_ids = list(candidate_papers_ids - processed_papers_ids - banned_paper_ids)
        depth += 1
        if depth > depth_limit:
            break
    return papers_metadata


def save_papers_metadata_to_db(_papers_metadata: list[PaperMetadata]) -> None:
    papers_metadata_df = pd.DataFrame([metadata.model_dump() for metadata in _papers_metadata])
    with sqlite3.connect(PAPERS_METADATA_DB_PATH) as conn:
        papers_metadata_df.to_sql("papers_metadata", con=conn, if_exists="append", index=False)


def read_papers_metadata_from_db() -> pd.DataFrame:
    try:
        with sqlite3.connect(PAPERS_METADATA_DB_PATH) as conn:
            papers_metadata_df = pd.read_sql_query("SELECT * FROM papers_metadata;", con=conn)
    except pd.errors.DatabaseError as e:
        if "no such table: papers_metadata" in str(e):
            # when starting from scratch
            papers_metadata_df = pd.DataFrame(columns=list(PaperMetadata.model_fields))
        else:
            raise
    return papers_metadata_df


if __name__ == "__main__":
    leandojo_paper_id = "87875a07976c26f82705de1fc70041169e5d652b"

    # 6a4501fefaf73261dc180ff86b52208679f3fb9c - A Survey on Deep Learning for Theorem Proving
    # 5ee871537ae51e7e2e93d2a70fff5d100649a655 - Mathematical Language Models: A Survey

    # we'd like to avoid surveys as they over-broaden retrieval space (especially highly cited ones)
    banned_words = ["survey", "overview"]

    # not relevant for us here, but highly-cited
    banned_paper_ids = {
        "411114f989a3d1083d90afd265103132fee94ebe",  # Mixtral of Experts (references Llemma in list of literature, but never in main teext, weird)
        "4c8cc2383cec93bd9ea0758692f01b98a035215b",  # UltraFeedback: Boosting Language Models with High-quality Feedback
        "560c6f24c335c2dd27be0cfa50dbdbb50a9e4bfd",  # TinyLlama: An Open-Source Small Language Model
    }

    papers_metadata = get_citations_graph(leandojo_paper_id, banned_words, banned_paper_ids, depth_limit=1)
