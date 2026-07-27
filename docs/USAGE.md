# Usage Guide

A practical walkthrough: set up once, then train, serve and test the detector.

## 1. Setup

Run once after cloning.

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

On macOS, XGBoost needs the OpenMP runtime:

```bash
brew install libomp
```

Without it the system still works — it logs a warning and trains the remaining models.

Every command below uses `.venv/bin/python` so you never depend on the system Python. To drop the
prefix for a terminal session, run `source .venv/bin/activate` first.

## 2. Prepare the data

```bash
.venv/bin/python scripts/download_data.py --rows 50000
```

Writes `data/raw/flows.csv`. With no `--url` it synthesises labelled traffic; to use a real capture
instead, pass `--url https://.../flows.csv` or drop your own CSV at that path.

## 3. Train the model

```bash
.venv/bin/python scripts/run_pipeline.py --rows 50000
```

Takes about a minute. It cleans the data, engineers features, selects the top 30, trains Random
Forest, XGBoost and Logistic Regression, picks the best on validation, and prints a summary table.

Produces:

| Path                                    | Contents                                  |
| --------------------------------------- | ----------------------------------------- |
| `models/best_model.pkl`                 | The model the API serves                  |
| `models/<name>_<version>.pkl`           | Every trained version, with metadata JSON |
| `data/processed/preprocessor.pkl`       | Fitted encoder and scaler                 |
| `reports/pipeline_summary.json`         | Metrics and model comparison              |
| `reports/*.png`                         | Confusion matrix, ROC curves, importances |

Add `--tune` for a hyperparameter grid search (slower), or `--skip-selection` to train on all
features.

## 4. Start the detection service

```bash
.venv/bin/uvicorn src.api.main:app --port 8000
```

Leave it running. Interactive documentation is at **http://localhost:8000/docs** — you can submit a
flow and read the verdict directly in the browser.

## 5. See it detect

In a second terminal, replay real labelled flows through the running service:

```bash
.venv/bin/python scripts/demo_detect.py --per-class 3
```

```
TRUE    PREDICTED   CONFIDENCE   SEVERITY   RESULT
dos     dos         1.0000       CRITICAL   correct
normal  normal      1.0000       -          correct
probe   probe       0.9999       CRITICAL   correct
r2l     r2l         0.9998       CRITICAL   correct
u2r     u2r         0.9968       CRITICAL   correct
```

Watch the first terminal at the same time: alerts appear in colour as they fire, and the throttler
starts suppressing once the same attack type exceeds five alerts a minute.

## 6. Query the API directly

| Method | Path            | Purpose                             |
| ------ | --------------- | ----------------------------------- |
| POST   | `/detect`       | Classify one flow                    |
| POST   | `/detect/batch` | Classify many flows at once          |
| GET    | `/detect/stats` | Rolling counters and recent alerts   |
| GET    | `/health`       | Status and which model is loaded     |
| GET    | `/metrics`      | Prometheus-style counters            |

```bash
curl -s http://localhost:8000/health
```

```bash
curl -s http://localhost:8000/detect/stats | python3 -m json.tool
```

A single detection returns the predicted class, confidence, whether it counts as an attack, its
severity, and `defaulted_features`.

**Read `defaulted_features` carefully.** It counts the model inputs your request left out, which the
system filled with values learned during training. A high count means the verdict is mostly based on
assumptions rather than your data — sparse flows are usually classified as normal because the
error-rate and connection-count features carry most of the signal. Send complete flow records when
the answer matters.

## 7. Where the alerts go

Configured under `alerting.channels` in `config/config.yaml`:

- **console** — colourised, in the terminal running the service
- **file** — one JSON object per line in `logs/alerts.log`
- **webhook** — HTTP POST to `alerting.webhook_url` (skipped if the URL is blank)

```bash
tail -f logs/alerts.log
```

## 8. Evaluate a saved model

```bash
.venv/bin/python scripts/evaluate_model.py --dataset data/raw/flows.csv
```

Prints the full per-class report and regenerates the plots in `reports/`.

## 9. Run the tests

```bash
.venv/bin/python -m pytest
```

```bash
.venv/bin/python -m pytest --cov=src --cov-report=term-missing
```

96 tests, 89% coverage.

## 10. Explore the data

```bash
.venv/bin/jupyter notebook notebooks/01_eda.ipynb
```

Class balance, correlation heatmap, feature distributions per class, and a PCA projection. Rebuild
it from source with:

```bash
.venv/bin/python scripts/make_notebook.py
```

## Changing settings

Edit `config/config.yaml` — split ratios, feature count, model hyperparameters, confidence
threshold, severity bands, alert channels, API port. Nothing needs recompiling; retrain only if you
changed something that affects training.

Any value can be overridden per run with `NIDS_<SECTION>__<KEY>`:

```bash
NIDS_API__PORT=9000 NIDS_DETECTION__CONFIDENCE_THRESHOLD=0.85 .venv/bin/uvicorn src.api.main:app
```

## Troubleshooting

| Symptom                                       | Cause and fix                                                            |
| --------------------------------------------- | ------------------------------------------------------------------------ |
| `ModuleNotFoundError: sklearn`                | Using the system Python. Prefix with `.venv/bin/` or activate the venv.   |
| Health reports `degraded`, detection 422s     | No trained model. Run `scripts/run_pipeline.py`.                          |
| `XGBoostError: libxgboost.dylib`              | Missing OpenMP. `brew install libomp`, or let it train the other models.  |
| Everything comes back `normal`                | Check `defaulted_features` — the request is too sparse to judge.          |
| `Address already in use`                      | Port 8000 is taken. Use `--port 8001`.                                    |
| Alerts stop mid-flood                         | Working as intended: the throttler caps repeats. Tune it in the config.   |
