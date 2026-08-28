# Evaluation protocol

## Cohort and inputs

- 242 training slides, 14 validation slides, and 16 untouched test slides.
- Seven organs; validation contains two slides per organ.
- Split by slide/patient before fitting panels, gene programs, or model weights.
- H&E-derived UNI2 features and coordinates are visible at inference.
- Query-spot and neighboring measured GEX are hidden.
- Targets are library-normalized `log1p` expression in a fixed 17,068-gene order.

## Fixed panels

Panels are derived from training slides only:

- all 17,068 genes;
- top 50 and 200 genes by pooled log1p variance;
- top 50 and 200 genes by within-slide variance.

No held-out values may select genes, checkpoints, thresholds, or examples.

## Primary scores

The primary prediction score is macro gene-wise spatial PCC: compute PCC over
spots for every eligible gene within each slide, average genes within slide,
then average held-out slides. Report slide-bootstrap 95% confidence intervals.
RMSE and MAE are co-primary calibration/error measures.

Flattened spot-by-gene PCC and spot-profile PCC answer different questions and
remain secondary. They must never be presented as gene-wise spatial PCC.

## Structural scores

- co-expression agreement: PCC between target and prediction gene-correlation
  matrices on fixed panels;
- Moran agreement: PCC between per-gene target and prediction Moran's I;
- local and wide signed-gradient agreement on k=6 and k=18 graphs, scaled by
  training-only per-gene standard deviation;
- masked spatial SSIM on panels up to 256 genes;
- amplitude ratio and effective rank to detect smoothing and low-rank template
  reuse.

## Study arms

| Arm | Coordinates | Spatial image attention | Within | Between | Seeds |
|---|---:|---:|---:|---:|---:|
| Local UNI2 MLP | no | no | no | no | 1, 10 |
| Spatial image base | yes | yes | no | no | 10 |
| Spatial + within | yes | yes | yes | no | 10 |
| Spatial + between | yes | yes | no | yes | 10 |
| Full SpatEX | yes | yes | yes | yes | 1, 2, 10 |

All arms share preprocessing, gene order, decoder width, optimizer, loss,
training limit, validation schedule, and fixed panels.

## Baseline disclosure

The local UNI2 MLP is the main fair baseline because it uses the same frozen
features and training cohort without spatial exchange. A frozen UNI2 ridge/PCA
probe is a useful capacity control.

Pretrained STPath and OmiCLIP are contextual baselines, not independent held-out
comparisons for these HEST-1K slides: both use large histology-ST pretraining
collections that include or substantially overlap HEST-1K. Exact STPath also
requires its GigaPath features; substituting UNI2 would change the method.

## Reporting rules

- Use validation only for development; show the 16-slide test set once at the end.
- Report mean, 95% CI, all organs, and all seeds rather than only best examples.
- Use identical target/prediction color limits for every gene-slide-method panel.
- Include one preregistered success and one failure example.
- Label any pretraining overlap and all unavailable-image spots.
