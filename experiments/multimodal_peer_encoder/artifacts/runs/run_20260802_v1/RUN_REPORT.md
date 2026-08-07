# Historical continuation result

Status: **STOP**. The predeclared historical continuation gate failed, so no
final checkpoints were fitted and no modern target was accessed.

## Pre-results boundary

- Protocol commit: `redacted-for-anonymous-release`
- Protocol tag: `multimodal-peer-protocol-v1-pre-results`
- Historical selection SHA-256:
  `e21dbc7350b1d339f771fbb8ec22ddda211f0a89cb2d55ba8394e5da230f0218`
- Historical cache-manifest SHA-256:
  `0e32965b0d33f59b25243733b4dd687a29fcb4497dba8eb86e931f9fae131fd0`
- Pre-fit and post-selection execution snapshot SHA-256:
  `5c3ba66f6351fa15b7311b09956255a2aa97357f70f6eaa4dd251eaceacb6604`

The cache contained 17,047 eligible feature-window rows derived from 6,169
unique exact filing texts. The 12 aligned training/target transitions contained
17,039 rows. All 18 required selection fits completed for 40 epochs.

## Gate result

| Width | Mean validation advantage | Slot wins | Decision |
|---:|---:|---:|---|
| 32 | -0.032612 | 0/6 | Selected; gate failed |
| 64 | -0.033971 | 0/6 | Not selected |

The gate required mean advantage of at least +0.005 and at least four wins in
six validation slots.

For the selected width, the equal-slot comparator mean was 0.409495 and the
model mean was 0.376883.

| Validation transition | Frozen blend | Width-32 model | Advantage |
|---|---:|---:|---:|
| H7->H8 | 0.462787 | 0.420824 | -0.041963 |
| H8->H9 | 0.432101 | 0.411243 | -0.020858 |
| H9->H10 | 0.412613 | 0.375677 | -0.036936 |
| H10->H11 | 0.396883 | 0.355574 | -0.041309 |
| H11->H12 | 0.389708 | 0.352060 | -0.037649 |
| H12->H13 | 0.362878 | 0.345923 | -0.016955 |

The selected-width seed-level population standard deviation was between
0.000528 and 0.001074 across validation transitions. The uniformly negative
differences therefore do not look like an unlucky single seed.

## Consequence

The run wrote `STOP`. It did not create `LOCKED_MANIFEST.json`, checkpoints,
`MODERN_TARGET_ACCESSED.json`, or `modern_evaluation.json`. Under the declared
protocol, this model is not eligible for the paper or for evaluation on W1->W7.
The frozen ICAIF manuscript remains unchanged.

This result rejects this particular residual-gated architecture and training
protocol relative to the fixed rank blend on the historical panel. It does not
establish that multimodal peer retrieval is generally ineffective.
