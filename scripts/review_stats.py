#!/usr/bin/env python3
"""
============================================================
review_stats.py -- statistics added in response to the ARR
reviews (Submission 397, Oct 2026). Pure stdlib, no API.
============================================================

Computes from the released answer/scored files of the 90
temporal questions (gpt-4o-mini reference run):
  [1] answer rate, EM, and correctness conditional on answering
      (abstention rule identical to analyze_results.py [6])
  [2] exact McNemar on answering vs. abstaining for every
      temporal-vs-time-blind pair, Bonferroni x21
  [3] per-category EM with exact McNemar vs. A1 (pre-registered
      H1 is stated per category: trend and drift), Bonferroni x21
  [4] per-category correct counts for all seven systems

Usage (from code/):
  python scripts/review_stats.py outputs/results outputs/temporal_questions_90.jsonl
============================================================
"""
import json, math, sys
from pathlib import Path
rd = Path(sys.argv[1])
qpath = Path(sys.argv[2])
S = [("A1","vanilla_rag"),("B1","entity"),("B2","community"),("B3","hybrid"),
     ("C","entity_temporal"),("D","community_temporal"),("E","hybrid_temporal")]
def load(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def binom2(k,n):
    if n==0: return 1.0
    pmf=lambda x: math.comb(n,x)*0.5**n
    pk=pmf(k); return min(1.0,sum(pmf(x) for x in range(n+1) if pmf(x)<=pk+1e-12))
qs={q["qid"]:q for q in load(qpath)}
print("score_tol present:", sum(1 for q in qs.values() if "score_tol" in q))
em,ans,cat={}, {}, {}
for lab,s in S:
    sc=load(rd/f"temporal_questions_90_{s}_scored.jsonl")
    an=load(rd/f"temporal_questions_90_{s}_answers.jsonl")
    em[lab]={r["qid"]:bool(r.get("match")) for r in sc}
    ans[lab]={a["qid"]:("INSUFFICIENT" not in (a.get("answer") or "").upper()) for a in an}
    for r in sc: cat[r["qid"]]=r.get("category")
qids=sorted(em["A1"])
print("n",len(qids))
print("\n[1] answered / correct / correct&answered / correct&abstained / cond.acc")
for lab,_ in S:
    a=sum(ans[lab][q] for q in qids); c=sum(em[lab][q] for q in qids)
    ca=sum(1 for q in qids if em[lab][q] and ans[lab][q]); cn=c-ca
    print(f"  {lab:>2}: answered {a:2d} ({100*a/90:5.1f}%)  correct {c:2d} ({100*c/90:5.1f}%)  c&a {ca:2d}  c&abst {cn}  cond {('%.1f%%'%(100*ca/a)) if a else '--'}")
print("\n[2] exact McNemar on answer indicator (temporal vs time-blind), Bonferroni x21")
for t in ["C","D","E"]:
    for b in ["A1","B1","B2","B3"]:
        x=sum(1 for q in qids if ans[t][q] and not ans[b][q]); y=sum(1 for q in qids if ans[b][q] and not ans[t][q])
        p=binom2(x,x+y); print(f"  {t} vs {b}: b={x:2d} c={y:2d} p={p:.3g} p_bonf={min(1,21*p):.3g}")
print("\n[3] H1 per category: EM, exact McNemar, Bonferroni x21")
for t in ["D","E","C"]:
    for c in ["trend","drift","latency"]:
        Q=[q for q in qids if cat[q]==c]
        mt=sum(em[t][q] for q in Q); ma=sum(em["A1"][q] for q in Q)
        x=sum(1 for q in Q if em[t][q] and not em["A1"][q]); y=sum(1 for q in Q if em["A1"][q] and not em[t][q])
        p=binom2(x,x+y)
        print(f"  {t} vs A1 {c:<8}: {mt}/30 vs {ma}/30  d={100*(mt-ma)/30:+.1f}pp  b={x} c={y} p={p:.3g} p_bonf={min(1,21*p):.3g}")
print("\n[4] best time-blind per category (max over A1-B3)")
for c in ["trend","drift","latency"]:
    Q=[q for q in qids if cat[q]==c]
    print(" ",c,{lab:sum(em[lab][q] for q in Q) for lab,_ in S})
