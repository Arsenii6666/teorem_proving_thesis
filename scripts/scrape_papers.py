import logging
from collections import deque
from datetime import date
from itertools import batched
from pathlib import Path
from typing import Annotated, Any, Final, cast
from tqdm.auto import tqdm

import pandas as pd
import requests
from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, PlainValidator, model_validator
from requests import Session
from requests.adapters import HTTPAdapter, Retry

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DATA_FOLDER: Final = Path(__file__).parent.parent / "data"
PAPERS_METADATA_PATH = DATA_FOLDER / "papers_metadata.csv"
ARXIV_ID_PATH: Final = DATA_FOLDER / "arxiv_ids.csv"
QUEUE_PATH: Final = DATA_FOLDER / "queue.txt"
VISITED_PATH: Final = DATA_FOLDER / "visited.txt"


# renamed: Paper -> PaperMetadata (to be more precise)
# switched: dataclasses -> pydantic (pydantic has field aliases)
## Semantic Scholar endpoint response is much richer than "citations", so fields have to be updated
## used datamodel-code-generator to generate fields from example json:
### datamodel-codegen --input data/lean_dojo.json --input-file-type json --output scripts/paper_metadata_model_autogen.py

# https://api.semanticscholar.org/api-docs/#tag/Paper-Data/operation/post_graph_get_papers

EXTRA_FIELDS: Final = [
    "title",
    "externalIds",
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
            ids: list[str] = [value for d in entry if (value := next(iter(d.values()))) is not None]
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

    @model_validator(mode="before")
    @classmethod
    def preprocess(cls, data: dict[str, Any]) -> dict[str, Any]:
        # extract arxiv_id
        if "externalIds" in data:
            external_ids: dict[str, str] = data["externalIds"]
            arxiv_id = external_ids.get("ArXiv")
            data.update({"arxiv_id": arxiv_id})

        # flatten openAccessPdf
        if "isOpenAccess" in data:
            if data["isOpenAccess"]:
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
def get_papers_metadata_from_semantic_scholar(paper_ids: list[str]) -> list[PaperMetadata]:
    batch_endpoint = "https://api.semanticscholar.org/graph/v1/paper/batch"
    with Session() as session:
        retries = Retry(
            total=15,
            backoff_factor=5.0,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False,
        )
        session.mount("http://", HTTPAdapter(max_retries=retries))
        session.mount("https://", HTTPAdapter(max_retries=retries))
        response = session.post(
            batch_endpoint,
            params={"fields": ",".join(EXTRA_FIELDS)},
            json={"ids": paper_ids},
            timeout=10,
        )
    if response.status_code == requests.codes.OK:
        data = response.json()
        papers_metadata = [PaperMetadata(**entry) for entry in data]
        return papers_metadata
    raise RuntimeError(f"Retrieval failed with code: {response.status_code}")


def get_citations_graph(origin_paper_id: str) -> list[PaperMetadata]:
    def _batch_process_queue(_queue: deque[str]) -> list[PaperMetadata]:
        # TODO: it is preferable to have dynamic batch size, due to citation limits
        batch_size = 20
        _papers_metadata: list[PaperMetadata] = []
        for batch in batched(tqdm(_queue), n=batch_size):
            _papers_metadata.extend(get_papers_metadata_from_semantic_scholar(list(batch)))
        return _papers_metadata

    papers_metadata: list[PaperMetadata] = []
    processed_papers_ids: set[str] = set()
    papers_ids_queue = deque([origin_paper_id])
    depth = 0
    while papers_ids_queue:
        logger.info(f"Processing {len(papers_ids_queue)} papers at depth {depth}")
        papers_metadata_batch = _batch_process_queue(papers_ids_queue)
        processed_papers_ids.update(
            paper_metadata.paper_id for paper_metadata in papers_metadata_batch
        )
        papers_metadata.extend(papers_metadata_batch)
        candidate_papers_ids = {
            citation_id
            for paper_metadata in papers_metadata_batch
            for citation_id in paper_metadata.citations_ids
        }
        new_papers_ids = candidate_papers_ids - processed_papers_ids - set(papers_ids_queue)
        papers_ids_queue = deque(new_papers_ids)
        depth += 1
    return papers_metadata


def save_papers_metadata(
    papers_metadata: list[PaperMetadata], filepath: Path = PAPERS_METADATA_PATH
) -> None:
    papers_metadata_df = pd.DataFrame([metadata.model_dump() for metadata in papers_metadata])
    papers_metadata_df.to_csv(filepath)


if __name__ == "__main__":
    origin_paper_id = "87875a07976c26f82705de1fc70041169e5d652b"  # LeanDojo
    papers_metadata = get_citations_graph(origin_paper_id)
    save_papers_metadata(papers_metadata)
