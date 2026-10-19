# Reproduce everything (one page)

Tested on Linux, Python 3.10, 2 cores, 3 GB RAM. Seed 42. Each command finishes in under 3 minutes on that machine.

## 0. Setup
1. `pip install -r requirements.txt` (pinned versions).
2. Put the Kaggle files `amazon_ecommerce_1M.csv`, `products.csv`, `reviews.csv` in the folder that contains `recsys-project/` (the raw data is not committed).
3. `cd recsys-project/src`

## 1. Data (Phase 1)
```
python make_manifest.py        # manifests/raw_manifest.json (SHA-256)
python clean.py                # processed tables + reports/table1_dataset_stats.md
python table1_full.py          # reports/table1_report_layout.md (Table I layout)
```

## 2. Split, metrics, baselines (Phase 2)
```
python split.py
python test_metrics.py         # hand-worked toy check: NDCG@5 = 0.7039
python baselines.py
python diagnose_signal.py      # reports/signal_diagnostics.md (1M file is random)
python make_control_data.py    # simulated positive control (labelled synthetic)
```
Then run `split.py` and `baselines.py` for the control prefix (see their `main(prefix=...)`).

## 3. Models and tuning, validation only (Phases 3 and 4)
```
python run_models.py control pop|knn|svd|als|bpr full
python run_models.py main    pop|knn|svd|als|bpr 5000 small
python content_amazon.py       # TF-IDF on real Amazon text (proxy task)
python hybrid.py               # hybrid, ablation, cold-item stress test
python e3_sweep.py             # E3 training-size sweep
```

## 4. Freeze and score the locked test once (Phase 5)
```
python freeze.py               # manifests/frozen_settings.json (commit BEFORE scoring)
python final_eval.py <dataset> <step>   # steps: random popularity itemknn svd als bpr content hybrid_adaptive cold finalize
python finalize.py             # Tables III and IV, bootstrap, E2/E5/E6/E7
```
`final_eval.py` refuses to overwrite a saved test result, so the test is scored once. To rerun it you must delete `reports/final_*_*/` deliberately.

## 5. Write-up (Phase 6)
```
python build_figures.py        # reports/figures/*.png
python build_report.py         # reports/final_report_sections.md (V to VII)
python assemble_report.py      # reports/final_report.md (full report)
python demo.py user CU42       # top-10 demo; also: demo.py product polo golf
streamlit run app.py           # browser UI (run from src/)
```

## Checks before submitting
- `grep -c TBD reports/final_report.md` prints 0.
- Every number in Section V is produced by `build_report.py` from `reports/*.md`.
- BPR varies by about 0.0002 NDCG@10 between runs (multi-threaded); everything else is deterministic.
