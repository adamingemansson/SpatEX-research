# Fixed biomarker panel

These genes are chosen before looking at the new study results. They provide
recognizable tissue biology and both epithelial/parenchymal and stromal views.

| Organ | Primary genes | Interpretation |
|---|---|---|
| Kidney | LRP2, UMOD | proximal tubule and thick ascending limb |
| Prostate | KLK3, ACPP | secretory luminal prostate programs |
| Bowel | FABP1, OLFM4 | differentiated enterocyte and crypt/stem compartment |
| Lung | SFTPC, DCN | alveolar type II and stromal matrix |
| Skin | KRT14, KRT10 | basal and differentiated epidermis |
| Brain | AQP4, MBP | astroglial and myelin-rich regions |
| Liver | ALB, GLUL | hepatocyte abundance and zonation |

Exploratory genes may be shown separately, for example CA9 in kidney,
COL1A1/ACTA2 for matrix/smooth muscle, and EPCAM/KRT8 for epithelium. They must
not replace the fixed panel after seeing performance.

For each primary gene, export target, prediction, and absolute error with the
same target/prediction scale. Add slide PCC, RMSE, target standard deviation,
prediction/target amplitude ratio, and the model name. A useful figure combines:

1. one slide where the pattern is recovered;
2. the paired organ slide where it is not;
3. local baseline versus full SpatEX;
4. a compact H&E thumbnail for stain and morphology context.
