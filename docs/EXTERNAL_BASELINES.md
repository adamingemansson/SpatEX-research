# External methods and comparison status

## Directly runnable and fair

- **Local UNI2 MLP:** same frozen image features, train split, output genes, and
  objective as SpatEX, but no coordinate or spot exchange. Included in the
  eight-arm study.
- **Frozen UNI2 PCA-ridge:** useful low-capacity probe. It should be trained only
  on the same 242 slides and evaluated with the same reports.

## Contextual pretrained baselines

- **STPath:** GigaPath patch embeddings plus geometry-aware transformer and
  pretrained generative objective. Its reported pretraining includes HEST-1K;
  therefore results on our HEST slides have cohort overlap. Exact execution is
  blocked until the official GigaPath cache is rebuilt; UNI2 cannot be inserted
  without changing the method.
- **OmiCLIP/Loki PredEx:** contrastively aligns histology and gene-expression
  representations and performs reference-weighted prediction. Its broad
  histology-ST pretraining and retrieval pool make it a context benchmark, not a
  like-for-like supervised baseline here.
- **GHIST:** incorporates morphology, cell-type, and neighborhood information at
  single-cell resolution. Its inputs and output resolution differ from Visium.

## Literature framing

The 2025 translational benchmark compares 11 methods across 28 metrics and shows
why ranking by one all-gene PCC is insufficient. The presentation follows its
use of multi-category metrics, marker maps, distribution plots, stain/domain
checks, and external-validation disclosure:

- [Translational benchmark](https://www.nature.com/articles/s41467-025-56618-y)
- [HEST-1K and HEST benchmark](https://proceedings.neurips.cc/paper_files/paper/2024/file/60a899cc31f763be0bde781a75e04458-Paper-Datasets_and_Benchmarks_Track.pdf)
- [STPath](https://pmc.ncbi.nlm.nih.gov/articles/PMC12618518/)
- [OmiCLIP](https://pmc.ncbi.nlm.nih.gov/articles/PMC12240810/)
- [GHIST](https://www.nature.com/articles/s41592-025-02795-z)
- [SEQUOIA spatial validation](https://pmc.ncbi.nlm.nih.gov/articles/PMC11564640/)
- [Hist2ST](https://academic.oup.com/bib/article/23/5/bbac297/6645485)
- [TRIPLEX](https://openaccess.thecvf.com/content/CVPR2024/papers/Chung_Accurate_Spatial_Gene_Expression_Prediction_by_Integrating_Multi-Resolution_Features_CVPR_2024_paper.pdf)

Cross-paper metric values should not be placed in one numeric leaderboard unless
the cohort, target normalization, panel, split, and aggregation are identical.
