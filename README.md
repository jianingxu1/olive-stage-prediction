# zrive-ds

Olive phenological stage prediction for Komorebi, built as a team project in the Zrive Applied Data Science program (Q1 2024).

## Problem statement

Olive growers need to anticipate how quickly fruit will mature to plan harvest timing. Given a parcel's current phenological stage and recent observations, we predicted **how many stages it would advance in the next 14 days** (classes 0–3). The data came from parcel observations and meteorological measurements across harvest seasons.

## Solution and performance

- Cleaned parcel and weather data, resolved inconsistent parcel IDs and stage records, and aligned each observation with a reading approximately 14 days later (±2 days) to create the target.
- Built numerical features from the current stage, week of year, time spent in the current and prior stages, and rolling weather statistics over 30, 90, and 365 days.
- Trained on the **2018–2021 harvests** and evaluated on the **2022 harvest** so the test set represented a later season. Compared a stage-based baseline with ridge logistic regression, random forest, and LightGBM.
- **LightGBM reached approximately 74% accuracy and 0.29 MSE**, versus approximately 66% accuracy and 0.46 MSE for the baseline. These are estimates read from the [project presentation](#project-materials), not exact values exported from an experiment log. Random forest also beat the baseline; ridge logistic regression did not.

The result is strongest for common outcomes. Recall for the rare three-stage advance class was very low, and the LightGBM train/test loss gap widened as more trees were added. The model would need further validation before use for harvest decisions.

## Options considered

| Choice | Decision |
| --- | --- |
| Predict the stage in two weeks, the number of stages advanced, or days until the next stage | Used **stages advanced** as a classification target so the output directly describes near-term maturation. The other formulations remain possible follow-ups. |
| Stage-based baseline, ridge logistic regression, random forest, LightGBM, XGBoost | The reported comparison covers the baseline, ridge logistic regression, random forest, and LightGBM. XGBoost was considered, but the presentation does not report a score for it. |
| Numerical and high-cardinality categorical features | Used numerical features for the initial models; left categorical features for later work. |

## Repository and data

`src/load.py` and `src/load_meteo.py` clean the parcel and weather data. `src/create_spine.py` creates the 14-day target. `src/models.py` contains the time split and baseline, and `src/metrics.py` calculates accuracy and MSE. Unit tests are in `tests/`; run them with `poetry install` and `poetry run pytest tests` using Python 3.11.

The underlying datasets and trained model artifacts are not included. This repository does **not** contain the full feature engineering and model training workflow needed to reproduce the presentation's scores.

## Project materials

The [Q1 2024 project presentation](docs/olive-stage-prediction-presentation.pdf) documents the experiments and results.
