import requests
import time
from collections import deque
import csv
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
                arxiv_id=citation.get('arxivId'),
                authors=citation.get('authors', []),
                doi=citation.get('doi'),
                intent=citation.get('intent', []),
                is_influential=citation.get('isInfluential', False),
                paper_id=citation['paperId'],
                title=citation.get('title', ''),
                url=citation.get('url'),
                venue=citation.get('venue', ''),
                year=citation.get('year')
            ) for citation in citations_data
        ]
        return citations
    elif response.status_code == 429:
        print(f"Rate limit exceeded for Paper ID: {paper_id}. Exiting.")
        return []
    else:
        print(f"Error: {response.status_code} for Paper ID: {paper_id}")
        return []

def save_queue(queue, filename=QUEUE_PATH):
    with open(filename, 'w') as file:
        for paper_id, depth in queue:
            file.write(f"{paper_id},{depth}\n")

def load_queue(filename=QUEUE_PATH):
    if filename.exists():
        with open(filename, 'r') as file:
            queue = deque()
            for line in file:
                paper_id, depth = line.strip().split(',')
                queue.append((paper_id, int(depth)))
            return queue
    return deque()

def save_visited(visited_papers, arxiv_ids, filename=VISITED_PATH):
    with open(filename, 'w') as file:
        for paper_id in visited_papers:
            file.write(f"{paper_id}\n")
        file.write("---\n")
        for arxiv_id in arxiv_ids:
            file.write(f"{arxiv_id}\n")

def load_visited(filename=VISITED_PATH):
    visited_papers = set()
    arxiv_ids = set()
    if filename.exists():
        with open(filename, 'r') as file:
            lines = file.readlines()
            separator = lines.index('---\n')
            visited_papers = set(line.strip() for line in lines[:separator])
            arxiv_ids = set(line.strip() for line in lines[separator+1:])
    return visited_papers, arxiv_ids

def find_all_citations(paper_id: str, delay, queue_file=QUEUE_PATH, visited_file=VISITED_PATH):
    papers = []

    queue = load_queue(queue_file)
    visited_papers, arxiv_ids = load_visited(visited_file)

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
            print(f"Depth: {depth}, Unique ArXiv IDs found: {len(arxiv_ids)}")
            for citation in citations:
                citation_id = citation.paper_id
                if citation.arxiv_id:
                    arxiv_ids.add(citation.arxiv_id)
                papers.append(citation)
                if citation_id not in visited_papers:
                    queue.append((citation_id, depth + 1))
            time.sleep(delay)
    except KeyboardInterrupt:
        print("\nProcess interrupted. Progress saved.")
        save_queue(queue, queue_file)
        save_visited(visited_papers, arxiv_ids, visited_file)
        return papers

    save_queue(queue, queue_file)
    save_visited(visited_papers, arxiv_ids, visited_file)
    return papers

def save_papers_to_csv(papers: list[Paper], filename=ARXIV_ID_PATH):
    if not filename.exists():
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["arxiv_id", "authors", "doi", "intent", "is_influential", "paper_id", "title", "url", "venue", "year"])
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        for paper in papers:
            authors = ", ".join([author['name'] for author in paper.authors])
            writer.writerow([paper.arxiv_id, authors, paper.doi, "; ".join(paper.intent), paper.is_influential, paper.paper_id, paper.title, paper.url, paper.venue, paper.year])

def load_existing_papers(filename=ARXIV_ID_PATH):
    existing_papers = set()
    if filename.exists():
        with open(filename, mode="r") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                existing_papers.add(row[5])
    return existing_papers

if __name__=="__main__":
    print(DATA_FOLDER)
    main_paper_id = "87875a07976c26f82705de1fc70041169e5d652b"
    existing_papers = load_existing_papers(ARXIV_ID_PATH)
    papers = find_all_citations(main_paper_id, delay=0.1)
    print(f"\nЗбереження інформації про {len(papers)} пейперів.")
    save_papers_to_csv(papers, ARXIV_ID_PATH)
    print("Дані збережено у файл 'data/arxiv_ids.csv'. Прогрес збережено.")

