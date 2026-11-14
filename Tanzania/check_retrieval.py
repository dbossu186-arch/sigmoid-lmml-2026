"""Independent review of retrieval results -> review.md.

For every question: top-3 documents with scores, an independent structural
check of the winner (status / exhibit / city / date / collection details parsed
from the document body, not just the title), and low-margin flags.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from retrieval_pipeline import (
    CITIES,
    TfidfIndex,
    load_corpus,
    load_questions,
    question_city,
    question_exhibits,
    score_documents,
)

LOW_MARGIN = 0.5  # combined-score gap between #1 and #2 below this is flagged


def body_facts(text: str) -> dict[str, str | None]:
    """Facts parsed from the body of a notice."""
    venue = re.search(r"reaches (.+?) in (Dar es Salaam|Arusha|Mwanza) on (\d+ \w+)", text)
    desk = re.search(r"collect the kit from the (.+?)\.", text)
    time = re.search(r"Collection begins at (\d\d:\d\d)", text)
    exhibit = re.search(r"reserved the (.+?) for the (.+?) stop", text)
    return {
        "venue": venue.group(1) if venue else None,
        "city": venue.group(2) if venue else None,
        "date": venue.group(3) if venue else None,
        "desk": desk.group(1) if desk else None,
        "time": time.group(1) if time else None,
        "body_exhibit": exhibit.group(1) if exhibit else None,
        "body_city": exhibit.group(2) if exhibit else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("corpus.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("questions.json"))
    parser.add_argument("--results", type=Path, default=Path("retrieved_ids.json"))
    parser.add_argument("--output", type=Path, default=Path("review.md"))
    args = parser.parse_args()

    documents = load_corpus(args.corpus)
    by_id = {doc.document_id: doc for doc in documents}
    questions = load_questions(args.questions)
    results = {r["question_id"]: r["document_id"] for r in json.loads(args.results.read_text())}
    exhibits = {doc.subject[0] for doc in documents if doc.subject[0]}
    index = TfidfIndex(documents)

    lines = ["# Retrieval review", ""]
    summary = {"ok": 0, "fail": 0, "low_margin": 0}
    rows = []
    for q in questions:
        text = q["question"]
        ranked = score_documents(documents, index, text)
        lex = {i: parts['lexical'] for _, i, parts in ranked}
        lexical_rank = sorted(range(len(documents)), key=lambda i: (-lex[i], i))
        chosen = results.get(q["question_id"])
        top = documents[ranked[0][1]]
        city = question_city(text)
        wanted = question_exhibits(text, exhibits)

        # Independent ground truth: the unique CURRENT notice whose *body* names
        # the asked exhibit and city.
        truth = [
            d.document_id
            for d in documents
            if d.status.startswith("CURRENT")
            and (f := body_facts(d.text))["body_exhibit"] in wanted
            and f["city"] == city
            and f["body_city"] == city
        ]
        facts = body_facts(by_id[chosen].text) if chosen in by_id else {}
        checks = {
            "matches pipeline top-1": chosen == top.document_id,
            "status CURRENT": by_id[chosen].status.startswith("CURRENT") if chosen in by_id else False,
            "exhibit matches": facts.get("body_exhibit") in wanted,
            "city matches (body)": facts.get("city") == city and facts.get("body_city") == city,
            "has desk + time": bool(facts.get("desk") and facts.get("time")),
            "unique structural answer": truth == [chosen],
        }
        margin = ranked[0][0] - ranked[1][0]
        ok = all(checks.values())
        summary["ok" if ok else "fail"] += 1
        low = margin < LOW_MARGIN
        summary["low_margin"] += low
        rows.append((q["question_id"], chosen, ok, margin))

        lines.append(f"## {q['question_id']} -> `{chosen}` {'OK' if ok else '**CHECK FAILED**'}"
                     f"{' **LOW MARGIN**' if low else ''}")
        lines.append(f"> {text}")
        lines.append("")
        lines.append(f"Asked: exhibit = {', '.join(sorted(wanted)) or '?'}; city = {city or '?'}")
        lines.append("")
        lines.append("| rank | document | score | lexical | title |")
        lines.append("|---|---|---|---|---|")
        for rank, (score, i, parts) in enumerate(ranked[:3], 1):
            d = documents[i]
            lines.append(f"| {rank} | {d.document_id} | {score:.3f} | {parts['lexical']:.3f} | {d.title} |")
        lex_top = [documents[i].document_id for i in lexical_rank[:3]]
        lines.append("")
        lines.append(f"Lexical-only top-3 (before reranking): {', '.join(lex_top)}")
        lines.append(
            f"Reason: `{chosen}` is the CURRENT ACCESS NOTICE for {facts.get('body_exhibit')} in "
            f"{facts.get('city')} ({facts.get('venue')}, {facts.get('date')}): collect from the "
            f"{facts.get('desk')} from {facts.get('time')}. Margin over #2: {margin:.3f}."
        )
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            lines.append(f"Failed checks: {', '.join(failed)}; structural truth: {truth}")
        lines.append("")

    head = [
        "# Retrieval review",
        "",
        f"Questions: {len(questions)}; all checks passed: {summary['ok']}; failed: {summary['fail']}; "
        f"low margin (< {LOW_MARGIN}): {summary['low_margin']}.",
        f"Smallest margin: {min(r[3] for r in rows):.3f}.",
        "",
        "| question | document | ok | margin |",
        "|---|---|---|---|",
        *[f"| {qid} | {doc} | {'yes' if ok else 'NO'} | {m:.3f} |" for qid, doc, ok, m in rows],
        "",
    ]
    args.output.write_text("\n".join(head + lines[2:]) + "\n", encoding="utf-8")
    print("\n".join(head[:4]))


if __name__ == "__main__":
    main()
