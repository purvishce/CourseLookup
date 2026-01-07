from .test import load_tests
from .eval import evaluate_all_retrieval
from pathlib import Path
import json
from collections import Counter

# Run full retrieval evaluation to ensure diagnostics are emitted
print("Running retrieval evaluation for all tests...")
mrrs = []
ndcgs = []
coverages = []
for test, result, progress in evaluate_all_retrieval():
    mrrs.append(result.mrr)
    ndcgs.append(result.ndcg)
    coverages.append(result.keyword_coverage)

avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0
avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0
avg_coverage = sum(coverages) / len(coverages) if coverages else 0

print(f"Mean Reciprocal Rank (MRR): {avg_mrr:.4f}")
print(f"Normalized DCG (nDCG): {avg_ndcg:.4f}")
print(f"Keyword Coverage: {avg_coverage:.1f}%")
print()

# Read diagnostics file
diag_path = Path(__file__).parent / "diagnostics" / "retrieval_diagnostics.jsonl"
if not diag_path.exists():
    print("No diagnostics file found at", diag_path)
    raise SystemExit(1)

missing_counter = Counter()
tests_with_missing = 0
total_tests = 0
examples = []

with open(diag_path, 'r', encoding='utf-8') as fh:
    for line in fh:
        total_tests += 1
        obj = json.loads(line)
        missing = obj.get('missing_keywords', [])
        if missing:
            tests_with_missing += 1
            for kw in missing:
                missing_counter[kw.lower()] += 1
            if len(examples) < 10:
                examples.append(obj)

print()
print(f"Total tests processed: {total_tests}")
print(f"Tests with missing keywords: {tests_with_missing} ({tests_with_missing/total_tests*100:.1f}%)")
print()
print("Top 20 missing keywords:")
for kw, cnt in missing_counter.most_common(20):
    print(f"  {kw}: {cnt}")

print()
print("Sample failing tests (up to 10):")
for ex in examples:
    print('-' * 80)
    print(f"Question: {ex.get('question')}")
    print(f"Keywords: {ex.get('keywords')}")
    print(f"Missing: {ex.get('missing_keywords')}")
    print("Top docs:")
    for d in ex.get('top_docs', [])[:5]:
        print(f"  Rank {d.get('rank')}: metadata={d.get('metadata')} snippet={d.get('snippet')[:200]!r}")

print('\nDone')
