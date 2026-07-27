# Model Report

## Experimental setup

| Item             | Value                                                        |
| ---------------- | ------------------------------------------------------------ |
| Dataset          | 50,000 labelled flows, KDD/NSL-KDD schema                     |
| Classes          | normal (53%), dos (30%), probe (11%), r2l (5%), u2r (1%)       |
| Split            | Stratified 70/15/15 — 34,999 train / 7,501 val / 7,500 test    |
| Features         | 41 raw + 22 engineered, reduced to 30 by mutual information    |
| Scaling          | StandardScaler after one-hot encoding                          |
| Validation       | 5-fold stratified cross-validation, macro F1                   |
| Selection metric | Validation macro F1                                            |

Macro averaging is used throughout because the classes are heavily imbalanced: a model that ignores
`u2r` entirely would still score above 0.99 on accuracy.

## Model comparison

| Model               | CV macro F1     | Validation macro F1 | Training time | Notes                              |
| ------------------- | --------------- | ------------------- | ------------- | ---------------------------------- |
| **XGBoost**         | **0.9883 ± 0.0030** | **0.9934**      | 3.3 s         | Selected — best on every metric     |
| Random Forest       | 0.9829 ± 0.0062 | 0.9844              | 5.4 s         | Close second, class-weighted        |
| Logistic Regression | 0.9822 ± 0.0042 | 0.9831              | 1.3 s         | Linear baseline, surprisingly solid |

Hyperparameters of the chosen model: 200 estimators, max depth 6, learning rate 0.1, `hist` tree
method, multi-class softprob objective, seed 42.

## Chosen model

XGBoost was selected on validation macro F1 (0.9934 against 0.9844 for Random Forest). It also has
the lowest cross-validation variance of the three candidates, trains fastest of the two tree
ensembles, and produces well-separated probability estimates — which matters here because the
alerting layer maps confidence directly onto severity. Gradient boosting suits this problem because
the discriminative signal lies in interactions between counters (high `count` *and* saturated
`serror_rate` means a flood; high `count` *and* high `rerror_rate` with low `same_srv_rate` means a
scan) rather than in any single feature.

## Held-out test performance

| Metric              | Score  |
| ------------------- | ------ |
| Accuracy            | 0.9993 |
| Precision (macro)   | 0.9966 |
| Recall (macro)      | 0.9952 |
| F1 (macro)          | 0.9959 |
| Precision (weighted)| 0.9993 |
| Recall (weighted)   | 0.9993 |
| F1 (weighted)       | 0.9993 |
| AUC-ROC (OvR macro) | 1.0000 |

### Per class

| Class  | Precision | Recall | F1     | Support |
| ------ | --------- | ------ | ------ | ------- |
| dos    | 1.0000    | 1.0000 | 1.0000 | 2,250   |
| normal | 0.9992    | 1.0000 | 0.9996 | 3,975   |
| probe  | 1.0000    | 1.0000 | 1.0000 | 825     |
| r2l    | 0.9973    | 0.9893 | 0.9933 | 375     |
| u2r    | 0.9867    | 0.9867 | 0.9867 | 75      |

### Confusion matrix

Rows are true classes, columns predicted.

|            | dos  | normal | probe | r2l | u2r |
| ---------- | ---- | ------ | ----- | --- | --- |
| **dos**    | 2250 | 0      | 0     | 0   | 0   |
| **normal** | 0    | 3975   | 0     | 0   | 0   |
| **probe**  | 0    | 0      | 825   | 0   | 0   |
| **r2l**    | 0    | 3      | 0     | 371 | 1   |
| **u2r**    | 0    | 0      | 0     | 1   | 74  |

Only five of 7,500 test flows are misclassified, and every error involves the two rare
privilege-abuse families. Three `r2l` flows are missed as normal — the operationally costly error,
since a missed remote-to-local attempt is a silent compromise. This is precisely why the alert
manager escalates `r2l` and `u2r` one severity level: the classifier is least confident exactly where
the impact is highest.

Plots are regenerated into `reports/` on every pipeline run: `xgboost_confusion_matrix.png`,
`xgboost_roc.png`, `xgboost_feature_importance.png`.

## Feature analysis

The top-ranked features after mutual-information selection are dominated by engineered rate and
ratio terms rather than raw counters:

`srv_count_ratio`, `mean_srv_interarrival`, `srv_packet_rate`, `packet_rate`, `mean_interarrival`,
`serror_rate`, `dst_host_srv_ratio`, `count`, `dst_bytes`, `interarrival_gap`, `error_rate_total`,
`log_duration`.

Feature engineering is therefore doing real work: seven of the twelve strongest signals do not exist
in the raw capture. The service-spread ratios (`srv_count_ratio`, `dst_host_srv_ratio`) separate
scans from floods, while the timing approximations separate slow interactive intrusions from
automated bursts.

## Limitations

**Synthetic evaluation.** These figures come from a dataset generated from per-class statistical
profiles. Real captures contain overlapping, mislabelled and adversarially crafted traffic, so
scores above 0.99 should be read as evidence that the pipeline is correct end to end, not as an
expected field accuracy. Validating against CICIDS2017 or UNSW-NB15 is the necessary next step.

**Rare-class support.** With 75 `u2r` flows in the test set, one additional error moves F1 by more
than a point. Confidence intervals on that row are wide.

**Known attacks only.** A supervised multiclass model cannot label a family it never saw during
training; a zero-day is most likely to be absorbed into the nearest known class or into `normal`.

**Feature-level input.** The system consumes flow records, not packets. A capture-to-flow stage
(Zeek, CICFlowMeter or similar) is required before it can sit on a live interface, and errors in
that stage propagate directly into detection.

**Static model.** Nothing retrains as traffic evolves. Concept drift will degrade performance
silently because no ground truth arrives at inference time.

## Future improvements

1. Validate and recalibrate on CICIDS2017 and UNSW-NB15, reporting per-dataset metrics side by side.
2. Add an unsupervised anomaly channel (Isolation Forest or an autoencoder) so novel traffic is
   flagged as anomalous rather than forced into a known class.
3. Attach SHAP explanations to each alert so an analyst sees which features drove the decision.
4. Add drift monitoring on the input distribution with a scheduled retraining trigger.
5. Calibrate probabilities (Platt scaling or isotonic regression) so severity thresholds mean the
   same thing across model versions.
6. Add adversarial robustness tests covering evasion by padding, fragmentation and timing jitter.
