## Approach
The baseline code compresses video by encoding the difference between each pixel and the same pixel in the prior frame, using a single arithmetic coding context for all pixels. My implementation improves on this in two ways: a better predictor and multiple coding contexts.

### Prediction
Instead of predicting each pixel purely from its temporal neighbor (same position in the prior frame), I blend use spatial and temporal prediction.

1. Spatial prediction: A PNG-style median predictor using already-encoded neighbors in the current frame — left, top, and top-left pixels. Specifically: `median(left, top, left + top - top_left)`. This exploits the fact that nearby pixels within the same frame tend to be very similar.

2. Temporal prediction: The same pixel position in the prior frame, which captures inter-frame redundancy in low-motion regions.

The final prediction is the average of the two: `(spatial + temporal) / 2`. The residual (actual − prediction, mod 256) is what gets arithmetic coded. Blending both predictors reduces the typical residual magnitude compared to either alone, since spatial prediction helps in high-motion regions where the prior frame is a poor match, and temporal prediction helps in smooth regions where spatial neighbors may be across an edge.

### Contexts
Rather than using a single probability model for all pixels, I use 8 contexts selected by local activity level, well within the 256-context limit. This allows smooth regions to build a tight probability distribution centered near zero without being diluted by the broader distribution of high-activity regions, and vice versa.

## Compression Ratios:
bourne.mp4 - 5.08
blueval.mp4 - 6.88
cocomelon.mp4 - 12.52
drewgooden.mp4 - 4.18