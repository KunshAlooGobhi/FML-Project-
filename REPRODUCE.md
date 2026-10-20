# Reproduce everything (one page)

Tested on Linux, Python 3.10, 2 cores, 3 GB RAM (and Windows, Python 3.11). Seed 42. Each command finishes in under 3 minutes on that machine.

## 0. Setup
1. `pip install -r requirements.txt` (pinned versions).
2. Put these files in the folder that contains `recsys-project/` (raw data is not committed):
   - Amazon Beauty 5-core: `reviews_Beauty_5.json.gz` and `meta_Beauty.json.gz` (Amazon product data, McAuley et al.; the SNAP/UCSD mirrors host them)
   - Kaggle sanity-check files: `amazon_ecommerce_1M.csv`, `products.csv`, `reviews.csv`
3. `cd recsys-project/src`

## 1. Real dataset: Amazon Beauty (headline)
```
python prepare_beauty.py       # k=5 core, ids, item text -> data/processed/*beauty*, reports/table1_beauty.md
python diagnose_beauty.py      # four signal checks (reports/signal_diagnostics_beauty.md)
python -c "import split; split.main('interactions_beauty', 'beauty_')"   # time split 70/15/15 (reports/split_summary_beauty.md)
python -c "import baselines; baselines.main('beauty_')"                  # validation baselines
for m in pop knn svd als bpr; do python run_models.py beauty_ $m 0 full; python run_models.py beauty_ $m 0 ext; done
python run_models.py beauty_ summary
python hybrid_beauty.py grid && python hybrid_beauty.py abl   # content grid, blend weight, ablation, new-item slice
python e3_sweep.py beauty
```

## 2. Sanity-check datasets
```
python make_manifest.py && python clean.py && python table1_full.py   # 1M file, Table I layout
python split.py && python baselines.py && python diagnose_signal.py   # 1M: random-like (signal_diagnostics.md)
python make_control_data.py    # simulated positive control (labelled synthetic); then split.main(src, 'control_') and baselines.main('control_') as above
python run_models.py control pop|knn|svd|als|bpr full ; python run_models.py main pop|knn|svd|als|bpr 5000 small
python content_amazon.py ; python hybrid.py ; python e3_sweep.py control
```

## 3. Freeze and score the locked test once
```
python freeze.py               # manifests/frozen_settings.json (commit BEFORE scoring)
python final_eval.py beauty val  random popularity itemknn svd als bpr content hybrid_adaptive cold finalize   # dry run on validation
python final_eval.py beauty test random popularity itemknn svd als bpr content hybrid_adaptive cold finalize   # locked test, once
# the same two commands with `control` or `main` in place of `beauty` for the sanity-check datasets; `finalize` writes Tables III and IV, bootstrap, E2/E5/E6/E7
```
`final_eval.py` refuses to overwrite a saved test result, so the test is scored once. To rerun you must delete `reports/final_*_*/` deliberately.

## 4. Write-up and demo
```
python build_figures.py        # reports/figures/*.png
python build_report.py         # reports/final_report_sections.md (V to VII)
python assemble_report.py      # reports/final_report.md; docx: pandoc final_report.md -o final_report.docx
python demo.py user            # terminal top-10 on real Beauty shoppers; demo.py product mascara
streamlit run app.py           # browser UI (run from src/)
```

## Checks before submitting
- `grep -c TBD reports/final_report.md` prints 0.
- Every number in Section V is produced by `build_report.py` from `reports/*.md`.
- BPR varies by about 0.0002 NDCG@10 between runs (multi-threaded); everything else is deterministic.
