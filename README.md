# MLB Hitter OBP Forecasting — Toronto Blue Jays Questionnaire, Question 5

**Applicant:** Marek Trawicki, Ph.D.  
**Python research script:** `TrawickiBlueJaysResearch.py`  
**Prepared:** November 22, 2026

## Research question

Among returning MLB hitters with at least 200 **approximate** plate appearances in consecutive seasons, does fixed-penalty ridge regression predict next-season on-base percentage (OBP) more accurately than a current-season empirical-shrinkage baseline?

This is a reproducible, **AI-assisted** screening research example. It is not a production MLB projection system or a claim that the Python code was created as part of a previously validated independent project.

## Files in this repository

- `TrawickiBlueJaysResearch.py`: primary executable Python research sample, with section-level and function-level explanatory comments
- `requirements.txt`: Python dependencies
- `RunBlueJaysPythonSample.m`: optional MATLAB launcher that executes the Python program (MATLAB is **not** needed to run the Python script)
- `.gitignore`: excludes data, virtual environments, and generated outputs

The raw approximately 8 MB batting dataset is intentionally **not** distributed here. The script acquires public historical batting data automatically as needed.

## Requirements

Python 3.11+ and the packages `numpy`, `pandas`, `scikit-learn`, and `matplotlib`. The applicant's successful real-data run used Python 3.14.8 in an isolated Windows environment.

## Run from Python

```bash
python -m pip install -r requirements.txt
python TrawickiBlueJaysResearch.py --self-test
python TrawickiBlueJaysResearch.py --download --data-dir Data --output Results
```

The self-test generates synthetic data strictly for program validation; these results must **not** be interpreted as MLB findings. The real-data command downloads and caches `Data/Batting.csv` if absent, constructs player-season forecasting observations, evaluates chronological forecasts, and writes the reported results under `Results/`.

## Optional MATLAB launcher

Place `RunBlueJaysPythonSample.m` and `TrawickiBlueJaysResearch.py` together in your existing project's `Code` folder. Run the following in MATLAB:

```matlab
R = RunBlueJaysPythonSample("selftest");
disp(R)
R = RunBlueJaysPythonSample("realdata");
disp(R)
```

The launcher automatically reuses or creates an isolated CPython environment under the project's `Data` folder. It does not convert the Python code into MATLAB code. Its real-data results are stored in `Results/BlueJaysPython`.

## Data source and provenance

The study uses a publicly accessible mirror of the SABR Lahman Baseball Database (publisher: https://sabr.org/lahman-database/). The cached batting CSV in the applicant's verified run contained 128,598 records spanning 1871–2025, with SHA-256:

`007551e2fe3072aff396a8573de61dceabe14dbf8de20038c8b60e2abe16978f`

The checksum records the exact data bytes used; it does not independently establish the authenticity of the third-party mirror. The program generates `data_provenance.json` with the source, file size, record count, years, and checksum.

## Reported real-data results

The numbers below come from the applicant's successfully executed real-data Python research program before its version-free filename was adopted. The code rename changes documentation and file references, **not** the Python algorithm. The renamed file should still be run once more to confirm the complete local workflow.

| Measure | Reported result |
|---|---:|
| Evaluation target seasons | 2021–2025 |
| Evaluation player-seasons | 1,184 |
| Distinct players | 442 |
| Empirical-shrinkage baseline MAE | 0.02466 |
| Ridge regression MAE | 0.02348 |
| Baseline minus ridge MAE | +0.00118 |
| Relative reduction in MAE | approximately 4.79% |
| 95% player-cluster bootstrap interval | [+0.00063, +0.00174] |

The exported player-level predictions and annual summaries were independently checked for numerical agreement with these metrics. This is **not** the same as an independent end-to-end reconstruction from raw batting data.

## Outputs

- `forward_predictions.csv`: individual player-season forecasts and errors
- `metrics_by_year.csv`: target-season performance comparison
- `research_summary.txt`: pooled metrics and bootstrap uncertainty interval
- `forward_mae.png`: visual comparison of prediction errors
- `data_provenance.json`: source and data fingerprint

## Scientific limitations

The evaluation population includes only hitters with at least 200 *approximate* plate appearances in both the forecast-origin and following seasons. Selection based on future playing time prevents the results from serving as unconditional forecasts for all players eligible at the origin date. The player-cluster bootstrap holds fitted predictions fixed, and does not fully reflect season-level dependence or model-development uncertainty. The reduction in historical prediction error is not evidence of a causal treatment effect or of an operational improvement.

## AI disclosure

OpenAI ChatGPT (GPT-6) substantially assisted with designing, writing, debugging, documenting, and packaging this demonstration. The final questionnaire provides further details on suggested changes, fixes, and verification. The applicant retains responsibility for reviewing the code, its assumptions, and its reported results.
