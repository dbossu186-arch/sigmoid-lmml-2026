"""Repaired retrieval pipeline for the Tanzania agents task.

Each question asks for the *current access notice* of one exhibit in one city.
The corpus contains, for every (exhibit, city) pair, a CURRENT ACCESS NOTICE
plus an ARCHIVED BACKGROUND decoy with nearly identical wording, the same
exhibit in two other cities, and 600 generic venue bulletins.

Retrieval = whole-document TF-IDF (one shared vector space for queries and
documents, proper tokenisation, positive IDF, cosine similarity, argmax) plus
metadata-aware reranking: the document's city must match the city named in the
question, the exhibit name must match, and current notices are preferred over
archived / withdrawn background notices. Ties are broken deterministically by
corpus order and are reported by check_retrieval.py.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flag_utils import derive_flag, load_verification_codes

CITIES = ("Dar es Salaam", "Arusha", "Mwanza")
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from give in is it of on or "
    "should the their them they this to under what when where which who will "
    "with visitor visitors".split()
)
# Reranking weights (dominant over lexical similarity, which lies in [0, 1]).
CITY_MATCH_BONUS = 2.0
EXHIBIT_MATCH_BONUS = 2.0
CURRENT_NOTICE_BONUS = 1.0
ARCHIVED_PENALTY = 1.0


@dataclass(frozen=True)
class Document:
    """One searchable source document."""

    document_id: str
    title: str
    text: str

    @property
    def status(self) -> str:
        """Notice status taken from the title prefix (e.g. CURRENT ACCESS NOTICE)."""
        return self.title.split(":", 1)[0].strip().upper() if ":" in self.title else ""

    @property
    def subject(self) -> tuple[str | None, str | None]:
        """(exhibit, city) parsed from titles shaped 'STATUS: Exhibit, City'."""
        if ":" not in self.title:
            return None, None
        rest = self.title.split(":", 1)[1].strip()
        if ", " not in rest:
            return rest, None
        exhibit, city = rest.rsplit(", ", 1)
        return exhibit.strip(), city.strip()

    def cities(self) -> set[str]:
        """Cities mentioned in title or body."""
        full = f"{self.title}\n{self.text}"
        return {city for city in CITIES if city in full}


def load_corpus(corpus_path: Path) -> list[Document]:
    """Read source documents from a JSON Lines corpus."""
    documents: list[Document] = []
    with corpus_path.open(encoding="utf-8") as corpus_file:
        for line in corpus_file:
            if not line.strip():
                continue
            row = json.loads(line)
            documents.append(Document(row["document_id"], row["title"], row["text"]))
    return documents


def load_questions(questions_path: Path) -> list[dict[str, str]]:
    """Read the ordered retrieval questions."""
    with questions_path.open(encoding="utf-8") as questions_file:
        questions = json.load(questions_file)
    if not isinstance(questions, list):
        raise ValueError("questions.json must contain a list.")
    return questions


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics (keeping hyphenated words), drop stopwords."""
    return [
        token
        for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
        if token not in STOPWORDS
    ]


class TfidfIndex:
    """Whole-document TF-IDF index; queries and documents share one vocabulary."""

    def __init__(self, documents: list[Document]) -> None:
        tokenised = [tokenize(f"{doc.title}\n{doc.text}") for doc in documents]
        document_frequency: Counter[str] = Counter()
        for tokens in tokenised:
            document_frequency.update(set(tokens))
        self.vocabulary = {term: i for i, term in enumerate(sorted(document_frequency))}
        n_docs = len(documents)
        # Smoothed, strictly positive IDF: rare terms weigh more.
        self.idf = np.zeros(len(self.vocabulary))
        for term, index in self.vocabulary.items():
            self.idf[index] = math.log((1 + n_docs) / (1 + document_frequency[term])) + 1.0
        self.matrix = np.vstack([self._vectorise(tokens) for tokens in tokenised])

    def _vectorise(self, tokens: list[str]) -> np.ndarray:
        vector = np.zeros(len(self.vocabulary))
        for term, count in Counter(tokens).items():
            index = self.vocabulary.get(term)
            if index is not None:
                vector[index] = (1.0 + math.log(count)) * self.idf[index]
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def similarities(self, query: str) -> np.ndarray:
        """Cosine similarity of the query against every document."""
        return self.matrix @ self._vectorise(tokenize(query))


def question_city(question: str) -> str | None:
    """City named in the question (None if zero or several)."""
    found = [city for city in CITIES if city.lower() in question.lower()]
    return found[0] if len(found) == 1 else None


def question_exhibits(question: str, exhibits: set[str]) -> set[str]:
    """Known exhibit names (from corpus titles) mentioned in the question."""
    lowered = question.lower()
    return {exhibit for exhibit in exhibits if exhibit.lower() in lowered}


def score_documents(
    documents: list[Document], index: TfidfIndex, question: str
) -> list[tuple[float, int, dict[str, float]]]:
    """Return (score, doc_index, components) for every document, best first."""
    exhibits = {doc.subject[0] for doc in documents if doc.subject[0]}
    city = question_city(question)
    wanted_exhibits = question_exhibits(question, exhibits)
    lexical = index.similarities(question)
    scored = []
    for i, doc in enumerate(documents):
        doc_exhibit, doc_city = doc.subject
        parts = {"lexical": float(lexical[i])}
        if city is not None:
            matches_city = doc_city == city if doc_city else doc.cities() == {city}
            parts["city"] = CITY_MATCH_BONUS if matches_city else 0.0
        if wanted_exhibits:
            parts["exhibit"] = EXHIBIT_MATCH_BONUS if doc_exhibit in wanted_exhibits else 0.0
        if doc.status.startswith("CURRENT"):
            parts["status"] = CURRENT_NOTICE_BONUS
        elif doc.status.startswith(("ARCHIVED", "EXPIRED", "SUPERSEDED", "WITHDRAWN")):
            parts["status"] = -ARCHIVED_PENALTY
        scored.append((sum(parts.values()), i, parts))
    # Highest score first; ties fall back to corpus order (stable, deterministic).
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored


def retrieve_documents(
    documents: list[Document], questions: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Return one source document for every question, in question order."""
    index = TfidfIndex(documents)
    results: list[dict[str, str]] = []
    for question in questions:
        best_index = score_documents(documents, index, question["question"])[0][1]
        results.append(
            {
                "question_id": question["question_id"],
                "document_id": documents[best_index].document_id,
            }
        )
    return results


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("corpus.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("questions.json"))
    parser.add_argument("--output", type=Path, default=Path("retrieved_ids.json"))
    return parser.parse_args()


def main() -> None:
    """Run retrieval, save IDs, and print the derived flag."""
    arguments = parse_arguments()
    documents = load_corpus(arguments.corpus)
    questions = load_questions(arguments.questions)
    results = retrieve_documents(documents, questions)
    arguments.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    flag = derive_flag(results, questions, load_verification_codes(arguments.corpus))
    print(f"Saved {len(results)} results to {arguments.output}.")
    print(flag)


if __name__ == "__main__":
    main()
