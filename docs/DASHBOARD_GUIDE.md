# Dashboard guide

The dashboard presents verified classification evaluation, training-only threshold tradeoffs, independent customer segments and testable business recommendations. It retains the existing buildless HTML/CSS/JavaScript framework and navy/teal/coral design. No hosting account or JavaScript package installation is required.

## Start locally

Run from the repository root in PowerShell:

```powershell
# Recommended fresh environment
python -m venv .venv-m3
.\.venv-m3\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-m3\Scripts\python.exe -m experiments.milestone3.build
.\.venv-m3\Scripts\python.exe -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open `http://127.0.0.1:8000`. Stop the server with Ctrl+C. If using the existing local dependency directory instead, set `$env:PYTHONPATH="$PWD\.experiment_dependencies"` and replace the executable with `python`. Python 3.14 and the pinned scikit-learn stack match the historical model artifacts. A browser-only viewer needs no Python libraries after the report bundle has been generated.

Do not double-click `index.html`: browser file-origin restrictions may block JSON fetching. A different port is fine if 8000 is occupied. To run from outside the repository, pass its absolute path to `--directory`.

## Sections and controls

| Section | Meaning and interaction |
|---|---|
| Overview | Verified customer count, class counts/rate, primary KPI and business target |
| Model comparison | Four-model comparison and ROC/PR curves; download CSV; expand the F1 chart |
| Model details | Labeled dropdown changes confusion matrix, importance, permutation table and OOF threshold curve |
| Threshold slider | Shows an existing training-OOF grid row. Starts at the selected threshold on model change. Never uses test labels to tune or update final metrics |
| Segments | Counts, response counts/rates, income/spending/purchase/household profiles; K and silhouette |
| Decisions | Evidence-based suggestions explicitly framed as campaign hypotheses |
| Completed exploratory analysis | Expandable original EDA charts |
| Team / footer | Existing team identities, method documentation and source provenance |

The interface uses short section titles and compact result summaries. Method definitions, interpretation limits and selection caveats are available through labeled **i** buttons: hover or focus to view, click/tap to pin, press Escape to dismiss, or click outside to close a pinned tip. Table captions and confusion counts remain available to screen readers. The threshold slider announces its probability cutoff and training-OOF F1. Additional F1 and threshold charts are expandable. No destructive or irreversible actions are present.

Positive-class F1 is the primary KPI. CV F1 uses 0.50; held-out classification metrics use per-model frozen thresholds. AP is sklearn Average Precision, distinct from trapezoidal PR AUC. Confusion matrices have actual rows and predicted columns, with TN/FP/FN/TP labels. Segment response rates are historical descriptive summaries, not classifier performance.

## Sources and refresh

`dashboard.js` fetches `milestone3/dashboard_data.json`. Figure paths reference generated `milestone3/*.png` files. The CSV download is `milestone3/model_comparison.csv`; the complete source run list/hashes is in `milestone3/provenance.json`. Chart values and prose winners are computed from saved experiment predictions and verified fitted artifacts, never placeholder numbers.

`experiments/milestone3/sources.json` pins the exact Milestone 2 runs. To intentionally report compatible newer runs, edit the manifest and rerun the builder. It rejects mismatched source hashes, split IDs, row labels, OOF selection and saved model evidence. Retain the original manifest/report when presenting a different experiment generation.

The website is read-only: it does not fit models, upload records or perform customer-level scoring. It uses no external CDN and can be served as static files. No publication has been performed as part of the repository task.

## Troubleshooting

- **Results unavailable / HTTP 404:** run the report builder and confirm the server's document root is `docs`.
- **Invalid report bundle:** restore a valid generated JSON; do not manually insert scores. The page hides the evidence panels and displays recovery instructions.
- **Missing figure:** a readable warning replaces the broken image. Regenerate report assets or restore the missing original EDA file.
- **ModuleNotFoundError:** install root requirements in the active interpreter, or activate the existing local dependency path as shown above.
- **Incompatible sklearn artifact:** use the pinned historical stack rather than silently resaving a model under a different version.
- **Builder validation error:** inspect the named source mismatch. Do not bypass the check or overwrite historical artifacts.
- **Small display:** the menu collapses, charts stack and wide result tables scroll horizontally. Keyboard focus indicators, labeled inputs, alt text and a static evaluation report support accessibility.

Browser verification results and any execution limitations are recorded in [MILESTONE3_COMPLETION.md](MILESTONE3_COMPLETION.md).
