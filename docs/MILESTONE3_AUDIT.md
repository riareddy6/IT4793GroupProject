# Milestone 3 initial repository audit

Audit performed November 5, 2026, before implementation.

| Area | Existing implementation | Gap and justified action |
|---|---|---|
| Data | Raw TSV, Milestone 1 cleaning/EDA script and notebook, processed CSV | Preserve cleaning and engineered definitions; verify regenerated data in a temporary directory. |
| Classification | Four estimator entry points, JSON configurations, fold-local sklearn pipelines, persisted shared split | Preserve parameters, folds and thresholds. Restore ROC AUC to shared metrics and comparison reporting. |
| Evaluation | Saved probabilities, OOF probabilities, fold metrics, thresholds, reports, fitted models | Add one verified export with combined ROC/PR, labeled confusion matrices, threshold curves and provenance. Recompute metrics from predictions rather than transcribing historical tables. |
| Clustering | Independent standardized K-Means, K=2–8, profiles and assignments | Verify assignments and response counts. Task reference 17.2% conflicts with saved approximately 21.8%; document verified result. |
| Website | docs/index.html, buildless HTML/CSS/JS, mobile navigation | Page contains future charts/pending metrics and only three model descriptions. Retain framework and branding, replace placeholders with generated data and interactive model/OOF threshold controls. |
| Structure | Existing common utilities already separate loading, preprocessing, evaluation, persistence | Retain layout; extract shared plot saving, threshold selection and completed-run discovery. Replace dynamic pathlib import and duplicated constants with named settings. |
| Documentation | Milestone 1/2 narratives and reproducibility notes | README still says planning stage; function docstrings mostly absent or too short. Add architecture, evaluation, dashboard, business and complete function reference. |
| QA/dependencies | Artifact verification script; pinned model dependencies | No focused automated suite; seaborn absent although Milestone 1 imports it. Add declared full-workflow dependencies, focused unit/integration checks and documentation coverage checks. |

## Evaluation risks to retain explicitly

- CV F1 is measured at 0.50, whereas reported final test F1 uses training-OOF-selected thresholds. They are different policies.
- OOF threshold-development scores reuse the labels selecting the threshold and are optimistic. Test labels must never choose a threshold.
- The holdout has already been inspected; this extension must not tune to its leaderboard.
- Milestone 1 globally imputes Income, but modeling correctly restores original missing/sentinel cells before fold-local imputation.
- Fixed 2014 feature dates and the earlier 2026 birth-year cleaning convention differ. Preserve for historical comparability; pre-offer availability still needs domain confirmation.
- K-Means labels are descriptive and are not classifier inputs. Income profile means exclude original missing cells; clustering uses imputed values.
- Do not infer causal effects, profitability, significant model superiority or cluster stability from current evidence.

Historical experiment directories, failed-run evidence and the user's untracked Milestone 2 audit/support files will be preserved. New deliverables are additive; documentation edits to existing Python files must preserve behavior except explicitly documented validation/reporting fixes.
