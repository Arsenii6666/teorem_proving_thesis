import logging
import time
from collections import deque
from itertools import batched
from pathlib import Path
from typing import Final
from datetime import date

import pandas as pd
import requests
from pydantic import BaseModel, Field, model_validator

from typing import Any

logger = logging.getLogger(__name__)
DATA_FOLDER: Final = Path(__file__).parent.parent / "data"
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
    "openAccessPdf", # url, status
    "tldr", # model, text
]


class PaperMetadata(BaseModel):
    paper_id: str = Field(alias="paperId")
    title: str
    arxiv_id: str | None  # from externalIds
    authors_ids: list[str]
    citation_count: int = Field(alias="citationCount")
    reference_count: int = Field(alias="referenceCount")
    citations_ids: list[str]
    references_ids: list[str]
    publication_date: date | None = Field(alias="publicationDate")
    influential_citation_count: int = Field(alias="influentialCitationCount")
    is_open_access: bool = Field(alias="isOpenAccess")
    open_access_pdf_url: str | None
    open_access_pdf_status: str | None
    tldr_text: str | None

    @model_validator(mode="before")
    @classmethod
    def preprocess(cls, data: dict[str, Any]):
        # extract arxiv_id
        external_ids: dict[str, str] = data["externalIds"]
        arxiv_id = external_ids.get("ArXiv")

        # flatten authors
        authors: list[dict[str, str | None]] = data["authors"]
        authors_ids: list[str] = [author_id["authorId"] for author_id in authors if author_id["authorId"] is not None]

        # flatten citations
        citations: list[dict[str, str | None]] = data["citations"]
        citations_ids = [citation_id["paperId"] for citation_id in citations if citation_id["paperId"] is not None]

        # flatten references
        references: list[dict[str, str | None]] = data["references"]
        references_ids = [reference_id["paperId"] for reference_id in references if reference_id["paperId"] is not None]

        # flatten openAccessPdf
        if data["isOpenAccess"]:
            open_access_pdf: dict[str, str] = data["openAccessPdf"]
            open_access_pdf_url = open_access_pdf["url"]
            open_access_pdf_status = open_access_pdf["status"]
        else:
            open_access_pdf_url = None
            open_access_pdf_status = None

        # extract tldr text
        tldr: dict[str, str | None] | None = data["tldr"]
        tldr_text = tldr.get("text") if tldr is not None else None

        processed_fields = {"arxiv_id": arxiv_id,
                       "authors_ids": authors_ids,
                       "citations_ids": citations_ids,
                       "references_ids": references_ids,
                       "open_access_pdf_url": open_access_pdf_url,
                       "open_access_pdf_status": open_access_pdf_status,
                       "tldr_text": tldr_text,
                       }
        return data | processed_fields


def get_papers_metadata_from_semantic_scholar(paper_ids: list[str]) -> list[PaperMetadata]:
    batch_endpoint = "https://api.semanticscholar.org/graph/v1/paper/batch"

    # TODO: make a proper retry
    response = requests.post(batch_endpoint, params={"fields": ",".join(EXTRA_FIELDS)}, json={"ids": paper_ids}, timeout=10)
    if response.status_code == requests.codes.OK:
        data = response.json()
        papers_metadata = [PaperMetadata(**entry) for entry in data]
        return papers_metadata
    if response.status_code == requests.codes.TOO_MANY_REQUESTS:
        logger.error(f"Rate limit exceeded. Exiting.")
        return []
    logger.error(f"Error: {response.status_code}.")
    return []


def get_citations_graph(origin_paper_id: str) -> list[PaperMetadata]:
    def _batch_process_queue(_queue: deque[str]) -> list[PaperMetadata]:
        # TODO: it is preferable to have dynamic batch size, due to citation limits
        delay = 10
        batch_size = 20
        _papers_metadata: list[PaperMetadata] = []
        for batch in batched(_queue, n=batch_size):
            _papers_metadata.extend(get_papers_metadata_from_semantic_scholar(list(batch)))
            time.sleep(delay)
        return _papers_metadata

    papers_metadata: list[PaperMetadata] = []
    processed_papers_ids: set[str] = set()
    papers_ids_queue = deque([origin_paper_id])
    depth = 0
    while papers_ids_queue:
        papers_metadata_batch = _batch_process_queue(papers_ids_queue)
        processed_papers_ids.update(paper_metadata.paper_id for paper_metadata in papers_metadata_batch)
        papers_metadata.extend(papers_metadata_batch)
        candidate_papers_ids = {citation_id for paper_metadata in papers_metadata_batch for citation_id in paper_metadata.citations_ids}
        new_papers_ids = candidate_papers_ids.difference(processed_papers_ids).difference(papers_ids_queue)
        papers_ids_queue = deque(new_papers_ids)
        depth += 1
        logger.info(f"There are {len(papers_ids_queue)} unseen papers at the depth {depth}. Processing...")
        # TODO (debug):
        if depth > 1:
            break
    return papers_metadata


if __name__ == "__main__":
    origin_paper_id = "87875a07976c26f82705de1fc70041169e5d652b"  # LeanDojo
    papers_metadata = get_citations_graph(origin_paper_id)
    papers_metadata