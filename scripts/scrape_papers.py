import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
import requests

DATA_FOLDER: Final = Path(__file__).parent.parent / "data"
ARXIV_ID_PATH: Final = DATA_FOLDER / "arxiv_ids.csv"
QUEUE_PATH: Final = DATA_FOLDER / "queue.txt"
VISITED_PATH: Final = DATA_FOLDER / "visited.txt"


@dataclass
class Paper:
    paper_id: str
    arxiv_id: str | None
    authors: list[dict[str, str]]
    doi: str | None
    intent: list[str]
    is_influential: bool
    title: str
    url: str | None
    venue: str | None
    year: int | None


def get_citations_from_semantic_scholar(paper_id: str) -> list[Paper]:
    url = f"https://api.semanticscholar.org/v1/paper/{paper_id}"
    response = requests.get(url)
    if response.status_code == requests.codes.OK:
        data = response.json()
        citations_data = data.get("citations", [])
        citations = [
            Paper(
                paper_id=citation["paperId"],
                arxiv_id=citation.get("arxivId"),
                authors=citation.get("authors", []),
                doi=citation.get("doi"),
                intent=citation.get("intent", []),
                is_influential=citation.get("isInfluential", False),
                title=citation.get("title", ""),
                url=citation.get("url"),
                venue=citation.get("venue", ""),
                year=citation.get("year"),
            )
            for citation in citations_data
        ]
        return citations
    if response.status_code == requests.codes.TOO_MANY_REQUESTS:
        print(f"Rate limit exceeded for Paper ID: {paper_id}. Exiting.")
        return []
    print(f"Error: {response.status_code} for Paper ID: {paper_id}")
    return []


def save_queue(queue: deque[tuple[str, int]], filename: Path = QUEUE_PATH) -> None:
    with filename.open("w") as file:
        for paper_id, depth in queue:
            file.write(f"{paper_id},{depth}\n")


def load_queue(filename: Path = QUEUE_PATH) -> deque[tuple[str, int]]:
    queue: deque[tuple[str, int]] = deque()
    if filename.exists():
        with filename.open() as file:
            for line in file:
                paper_id, depth = line.strip().split(",")
                queue.append((paper_id, int(depth)))
    return queue


def save_visited(visited_papers: set[str], filename: Path = VISITED_PATH) -> None:
    with filename.open("w") as file:
        for paper_id in visited_papers:
            file.write(f"{paper_id}\n")


def load_visited(filename: Path = VISITED_PATH) -> set[str]:
    visited_papers = set()
    if filename.exists():
        with filename.open() as file:
            lines = file.readlines()
            visited_papers = {line.strip() for line in lines}
    return visited_papers


def find_all_citations(
    paper_id: str,
    delay: float,
    queue_file: Path = QUEUE_PATH,
    visited_file: Path = VISITED_PATH,
) -> list[Paper]:
    papers = []

    queue = load_queue(queue_file)
    visited_papers = load_visited(visited_file)

    if not queue:
        queue.append((paper_id, 0))

    try:
        while queue:
            current_paper_id, depth = queue.popleft()
            if current_paper_id in visited_papers:
                continue
            visited_papers.add(current_paper_id)
            citations = get_citations_from_semantic_scholar(current_paper_id)
            if citations is None:
                break
            print(f"Depth: {depth}, Unique ArXiv IDs found: {len(visited_papers)}")
            for citation in citations:
                citation_id = citation.paper_id
                papers.append(citation)
                if citation_id not in visited_papers:
                    queue.append((citation_id, depth + 1))
            time.sleep(delay)
    except KeyboardInterrupt:
        print("\nProcess interrupted. Progress saved.")
        save_queue(queue, queue_file)
        save_visited(visited_papers, visited_file)
        return papers

    save_queue(queue, queue_file)
    save_visited(visited_papers, visited_file)
    return papers


def save_papers_to_csv(papers: list[Paper], filename: Path = ARXIV_ID_PATH) -> None:
    papers_df = pd.DataFrame(papers)
    papers_df.to_csv(filename, mode="a")


if __name__ == "__main__":
    print(DATA_FOLDER)
    main_paper_id = "87875a07976c26f82705de1fc70041169e5d652b"
    papers = find_all_citations(main_paper_id, delay=0.1)
    print(f"\nSaving information about {len(papers)} papers.")
    save_papers_to_csv(papers, ARXIV_ID_PATH)
    print("Data saved to 'data/arxiv_ids.csv'. Progress saved.")
