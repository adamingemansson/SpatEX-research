# Model architecture

## SpatEX

```text
UNI2 features + coordinates
            |
   spatial conditioner
            |
     full-GEX decoder
       /           \
 within-gene    between-spot
       \           /
        learned gates
            |
     expression output
```

The within-gene branch refines co-expression through a centered gene-program
basis fitted on training slides. The between-spot branch attends to nearby
spots using image context, relative position, and the model's predicted
expression. Neither branch receives measured query expression.

## WAE variants

Both WAE variants add a latent residual to the same SpatEX prediction:

```text
SpatEX output + residual_decoder(image context, z)
```

- The standard prior is `N(0, I)`.
- The conditional prior is predicted from image context; its posterior uses
  FiLM conditioning.
- MMD matches the posterior to the chosen prior.
- Antithetic draw pairs average to the deterministic prediction.

Measured expression enters only the training posterior and the labelled
posterior-reconstruction diagnostic.

## Evaluation paths

- `deterministic`: SpatEX point prediction;
- `predictive_mean`: mean of H&E-only prior draws;
- `sample_0`: one prior draw;
- `posterior_reconstruction`: training-style diagnostic using measured GEX;
- `zero_latent`: prior-centre control;
- `shuffled_posterior`: latent-specificity control.

Deployable paths always record `query_gex_visible: false`.
