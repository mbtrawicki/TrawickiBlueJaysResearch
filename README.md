# MLB Hitter OBP Forecasting — Toronto Blue Jays Questionnaire, Question 5

**Applicant:** Marek Trawicki, Ph.D.  
**Python research script:** `TrawickiBlueJaysResearch.py`  
**Prepared:** November 22, 2026

## Research question

Among returning MLB hitters with at least 200 **approximate** plate appearances in consecutive seasons, does fixed-penalty ridge regression predict next-season on-base percentage (OBP) more accurately than a current-season empirical-shrinkage baseline?

This is a reproducible, **AI-assisted** research screening demonstration. It is not a production MLB projection system or a previously validated independent research project.

## Repository files

- `TrawickiBlueJaysResearch.py` — complete, documented, executable Python research code, including section-level and function-level comments
- `README.md` — research question, execution instructions, reported results, and limitations
- `requirements.txt` — Python dependencies
- `.gitignore` — instructions for Git to exclude local datasets, Python environments, and generated output files from future commits

The MATLAB launcher is intentionally **not** included: Python alone can run the analysis. The original batting dataset and generated output files are also excluded from this repository to minimize storage requirements.

## Requirements

Python 3.11 or later and the packages `numpy`, `pandas`, `scikit-learn`, and `matplotlib`. The applicant's successful MLB-data execution used Python 3.14.8 in an isolated Windows environment.

## Execute the research code

From the repository directory, run:

```bash
python -m pip install -r requirements.txt
python TrawickiBlueJaysResearch.py --self-test
python TrawickiBlueJaysResearch.py --download --data-dir Data --output Results
```

The self-test uses **synthetic data strictly for software testing**; it does not produce empirical MLB findings. The real-data command retrieves and caches `Data/Batting.csv` if absent, aggregates multiple team stints into player-seasons, builds features, evaluates next-season OBP forecasts chronologically, and writes results to `Results/`. No manual data download is necessary when the public source is reachable.

## Public batting data and reproducibility

The program retrieves a publicly accessible mirror of the SABR Lahman Baseball Database ([SABR publisher page](https://sabr.org/lahman-database/)). The applicant's verified local execution processed **128,598 batting rows covering 1871–2025**, with the cached CSV SHA-256 checksum:

`007551e2fe3072aff396a8573de61dceabe14dbf8de20038c8b60e2abe16978f`

The checksum documents the exact bytes processed, but by itself does not establish that a third-party mirror is an authentic copy of the publisher's release. The software exports `data_provenance.json` with the data location, row counts, historical coverage, file size, and checksum.

## Reported real-data results

The values below are from successfully completed real-MLB-data execution of the same computational workflow before the source was renamed `TrawickiBlueJaysResearch.py`. The filename and comments changed, but the statistical computations did not. A final local real-data run of the renamed program remains advisable.

| Measure | Result |
|---|---:|
| Evaluation target seasons | 2021–2025 |
| Evaluation player-seasons | 1,184 |
| Distinct evaluated players | 442 |
| Empirical-shrinkage baseline MAE | 0.02466 |
| Ridge-regression MAE | 0.02348 |
| Baseline minus ridge MAE | +0.00118 |
| Relative reduction in MAE | 4.79% |
| 95% player-cluster bootstrap interval for improvement | [+0.00063, +0.00174] |

The exported player-season predictions and annual summaries were independently audited for numerical consistency. This does **not** constitute an independent end-to-end reproduction from raw batting data.

## Output files

A real-data run writes the following files under the chosen `--output` directory:

- `forward_predictions.csv` — player-season forecasts and errors
- `metrics_by_year.csv` — annual MAE and RMSE comparison
- `research_summary.txt` — aggregate performance and bootstrap interval
- `forward_mae.png` — chart of annual prediction errors
- `data_provenance.json` — source information and checksum

These are generated artifacts, not files that must be uploaded to GitHub.

## Scientific limitations

The evaluation population includes only hitters who achieve at least 200 **approximate** plate appearances in both the forecast-origin and following seasons. Because next-season playing time contributes to inclusion, the reported accuracy comparison does not describe unconditional forecasting performance for every player eligible at prediction time. The player-cluster bootstrap treats fitted forecasts as fixed and does not fully account for shared season-level shocks or model-development uncertainty. A historical error reduction is not evidence of a causal effect or an operational improvement.

## AI-use disclosure

OpenAI ChatGPT (GPT-6) substantially assisted in developing, writing, documenting, debugging, and packaging this screening demonstration. The applicant executed the Python analysis on publicly available MLB batting data and remains responsible for reviewing the code, statistical assumptions, limitations, and results. The questionnaire provides further details of the AI assistance and validation process.
