"""Assemble reports/final_report.md: front matter (I-IV) + generated V-VI + status (VII) + conclusion (VIII) + references.
Run build_report.py and table1_full.py first. Fails if any [TBD] or @@ placeholder remains."""
import re
from pathlib import Path
R = Path(__file__).resolve().parent.parent / "reports"
front = (R / "report_front.md").read_text()
t1 = "\n".join(l for l in (R / "table1_beauty.md").read_text().splitlines() if l.startswith("|"))
t1 += "\n\nEvery user and item has at least five reviews (k = 5), a median of 6 reviews per user and 9 per item, and the data are 99.93% sparse."
front = front.replace("@@TABLE1@@", t1)
gen = (R / "final_report_sections.md").read_text()
gen = gen.split("\n", 2)[2]                                   # drop the generated-file banner line
body, concl = gen.split("## VII. Conclusion")
body = body.replace("### Does the adaptive hybrid hurt ordinary (non-new-item) recommendations?", "**Does the adaptive hybrid hurt ordinary (non-new-item) recommendations?**")
dup = "A model counts as an improvement only if its 95% interval excludes zero AND the Bonferroni-adjusted p < 0.05. With 1,000 resamples the smallest reportable p is below 1/1,000.\n\n"
body = body.replace(dup, "")
status = """## VII. Project Status

| Phase | Focus | Status |
|---|---|---|
| 1 | Data download, schema check, manifest, cleaning, Table I | Done (`src/make_manifest.py`, `clean.py`) |
| 2 | Baselines, split and evaluation harness | Done (`split.py`, `metrics.py`, `evaluate.py`, `baselines.py`) |
| 3 | Item-kNN, SVD, ALS/BPR, TF-IDF content model, tuning | Done (`run_models.py`, `content_amazon.py`) |
| 4 | Hybrid, cold-start rules, ablations (E2 to E4) | Done (`hybrid.py`, `e3_sweep.py`) |
| 5 | Freeze settings, score locked test once, bootstrap tests, E5 to E7 | Done (`freeze.py`, `final_eval.py`, `finalize.py`) |
| 6 | Results, discussion, limitations, figures, demo, viva notes | Done (`build_report.py`, `build_figures.py`, `demo.py`) |
| 7 | Real dataset (Amazon Beauty 5-core): rerun of phases 2 to 5 and report rebuild | Done (`prepare_beauty.py`, `hybrid_beauty.py`, `final_eval.py beauty`) |

Code, manifests, reports and figures are in the project repository; `REPRODUCE.md` lists the exact commands.
"""
refs = (R / "report_refs.md").read_text()
out = front + "\n" + body.rstrip() + "\n\n" + status + "\n## VIII. Conclusion" + concl.rstrip() + "\n\n" + refs
(R / "final_report.md").write_text(out)
bad = [m for m in re.findall(r"\[TBD\]|@@\w+@@", out)]
print("words:", len(out.split()), "| placeholders left:", bad)
assert not bad
