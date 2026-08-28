# Presentation plan

## Core claim

SpatEX asks whether explicitly modeling relationships within a spot's predicted
gene profile and between neighboring spots preserves biological structure that
gene-wise PCC alone misses. The presentation should establish that claim before
discussing WAE experiments.

## Suggested 11-slide deck

1. **Problem and hypothesis**  
   H&E-to-ST predictions can be smooth, low-rank, and locally displaced. Point
   metrics alone may reward plausible tissue templates.

2. **Data and leakage-safe protocol**  
   242/14/16 train/validation/test slides, seven organs, fixed 17,068 genes,
   normalized-log1p targets, H&E+coordinates only. Show a split/organ table and
   disclose foundation-model overlap.

3. **SpatEX overview**  
   Use the overview flow from `MODEL_FLOW.md`: UNI2, spatial conditioner,
   full-GEX decoder, parallel within/between refiners, learned gates.

4. **What within and between do**  
   Show the gene-program projection and neighbor-attention edge in detail. State
   that both operate on predicted GEX and never measured query GEX.

5. **Fair study design**  
   Table the five architecture states and seeds. Emphasize identical data,
   objective, budget, panels, and checkpoint rule.

6. **Primary benchmark**  
   Forest/dot plot of gene-wise spatial PCC and RMSE with 95% CIs for all genes,
   HVG-50/200, and within-slide-variance 50/200. Include local UNI2 first.

7. **Does each module do its intended job?**  
   Ablation deltas for co-expression, Moran agreement, local/wide gradient PCC,
   SSIM, amplitude ratio, and effective rank. Do not collapse these into one
   invented score.

8. **Robustness**  
   Organ-by-method heatmap, three-seed full-model distribution, and per-gene
   rank stability. Show stain distance as a diagnostic, not a causal claim.

9. **Preregistered marker maps**  
   Target/prediction/error rows for the fixed biomarkers. Use shared scales and
   include paired slides from the same organ when performance differs.

10. **Failure analysis**  
    Show low-amplitude/template reuse, blur-tolerant versus exact PCC, and one
    gene absent from a slide. Explain why low RMSE or high PCC alone can mislead.

11. **Conclusion and next experiment**  
    State whether within improves co-expression, between improves gradients,
    and full SpatEX improves held-out prediction reproducibly. Keep WAE as a
    negative/uncertainty result unless it adds calibrated information.

## Figures to export after evaluation

- cohort/split table;
- model flowchart and one detailed within/between panel;
- primary metric forest plot;
- structured-metric ablation plot;
- organ-by-method heatmap;
- seed robustness plot;
- fixed biomarker map grid;
- stain-distance versus PCC scatter;
- effective-rank/amplitude diagnostic;
- optional WAE deterministic-versus-sampled comparison.

Every figure should have a machine-readable TSV/JSON next to the PNG/PDF, use a
fixed method order, and record the evaluation report paths that produced it.
