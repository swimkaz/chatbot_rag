"""Tiny retrieval eval: for each question, did we find the right source file?

Run:  python eval.py [notes_folder]
Edit eval_set.json to add your own questions as you add notes.
"""
import json
import sys

from rag import Index, load_chunks

folder = sys.argv[1] if len(sys.argv) > 1 else "notes"
cases = json.load(open("eval_set.json"))
index = Index(load_chunks(folder))

hits = 0
for case in cases:
    found = [c.source for c, _ in index.search(case["question"], k=3)]
    ok = case["expected_source"] in found
    hits += ok
    print(f"{'PASS' if ok else 'FAIL'}  {case['question']}  -> {found}")
print(f"\nhit@3: {hits}/{len(cases)}")
