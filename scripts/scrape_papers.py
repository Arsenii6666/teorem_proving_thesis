import requests
import time
from collections import deque
import pandas as pd
from dataclasses import dataclass
from typing import Final
from pathlib import Path

DATA_FOLDER: Final = Path(__file__).parent.parent / "data"
ARXIV_ID_PATH: Final = DATA_FOLDER / "arxiv_ids.csv"
QUEUE_PATH: Final = DATA_FOLDER / "queue.txt"
VISITED_PATH: Final = DATA_FOLDER / "visited.txt"


@dataclass
class Paper:
    arxiv_id: str | None
    authors: list[dict]
    doi: str | None
    intent: list[str]
    is_influential: bool
    paper_id: str
    title: str
    url: str | None
    venue: str | None
    year: int | None


def get_citations_from_semantic_scholar(paper_id: str) -> list[Paper]:
    url = f"https://api.semanticscholar.org/v1/paper/{paper_id}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        citations_data = data.get("citations", [])
        citations = [
            Paper(
                arxiv_id=citation.get("arxivId"),
                authors=citation.get("authors", []),
                doi=citation.get("doi"),
                intent=citation.get("intent", []),
                is_influential=citation.get("isInfluential", False),
                paper_id=citation["paperId"],
                title=citation.get("title", ""),
                url=citation.get("url"),
                venue=citation.get("venue", ""),
                year=citation.get("year"),
            )
            for citation in citations_data
        ]
        return citations
    elif response.status_code == 429:
        print(f"Rate limit exceeded for Paper ID: {paper_id}. Exiting.")
        return []
    else:
        print(f"Error: {response.status_code} for Paper ID: {paper_id}")
        return []


def save_queue(queue, filename=QUEUE_PATH):
    with open(filename, "w") as file:
        for paper_id, depth in queue:
            file.write(f"{paper_id},{depth}\n")


def load_queue(filename=QUEUE_PATH):
    if filename.exists():
        with open(filename, "r") as file:
            queue = deque()
            for line in file:
                paper_id, depth = line.strip().split(",")
                queue.append((paper_id, int(depth)))
            return queue
    return deque()


def save_visited(visited_papers, filename=VISITED_PATH):
    with filename.open("w") as file:
        for paper_id in visited_papers:
            file.write(f"{paper_id}\n")


def load_visited(filename=VISITED_PATH):
    visited_papers = set()
    if filename.exists():
        with open(filename, "r") as file:
            lines = file.readlines()
            visited_papers = set(line.strip() for line in lines)
    return visited_papers


def find_all_citations(
    paper_id: str, delay, queue_file=QUEUE_PATH, visited_file=VISITED_PATH
):
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


def save_papers_to_csv(papers: list[Paper], filename=ARXIV_ID_PATH):
    df = pd.DataFrame(papers)
    df.to_csv(filename, mode="a")


if __name__ == "__main__":
    print(DATA_FOLDER)
    main_paper_id = "87875a07976c26f82705de1fc70041169e5d652b"
    papers = find_all_citations(main_paper_id, delay=0.1)
    print(f"\nЗбереження інформації про {len(papers)} пейперів.")
    save_papers_to_csv(papers, ARXIV_ID_PATH)
    print("Дані збережено у файл 'data/arxiv_ids.csv'. Прогрес збережено.")
