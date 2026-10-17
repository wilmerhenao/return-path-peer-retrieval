# Selected paper results

These four JSON files are aggregate-only extracts from saved experiments, not new runs. Each records the original relative artifact path, original SHA-256, and selected top-level fields. Values in those fields are copied exactly. `SHA256SUMS.txt` checks the public extract bytes; an original artifact's hash and its extract's hash are different identities.

| File | Scope |
| --- | --- |
| [full19_frozen_blend.json](full19_frozen_blend.json) | Fixed 80/20 blend, return path and TF-IDF on identical eligible queries/candidates across 19 transitions and the existing volatility regimes |
| [frozen_blend_depth_ticker.json](frozen_blend_depth_ticker.json) | Six modern transitions at k = 1, 5, 10, 20, plus the ticker-syntax exclusion with TF-IDF refitting and complete reranking |
| [learned_fusion.json](learned_fusion.json) | Fixed histogram gradient boosting: 12 historical training transitions and six modern evaluations |
| [cnn_historical_v1.json](cnn_historical_v1.json) | Paper CNN widths 32/64, seeds 11/29/47, six historical continuation slots and the failed continuation decision |

The blend configuration and weight were selected using H1→H2 through H12→H13 (12 transitions). H13→W1 and W1→W2 through W6→W7 are the seven transitions held out from that selection. Overall and regime summaries include selection periods. CNN continuation tests H7→H8 through H12→H13 occur after each CNN training fold but remain inside the blend's selection period; they are not an untouched test of the blend. The CNN experiment stopped before modern targets were accessed. These descriptive comparisons do not establish a cause of the nonlinear models' underperformance.

All reported means are unweighted means of transition-level scores. Across-transition standard deviations use population normalization (`ddof=0`) and are not inferential error bars. The summary verifier checks checksums, arithmetic, and the selection/held-out partitions using only Python's standard library:

```bash
python scripts/verify_paper_results.py
```

It does not rerun a model, verify the unavailable source artifacts, or certify the underlying data. To regenerate the extracts when the original experiment tree is available:

```bash
python scripts/export_paper_results.py --source-root /path/to/original/project --output-dir /tmp/paper-result-extracts
```

Raw returns, filing text, eligible ticker lists, dense vectors, checkpoints, and private execution records are omitted. The frozen-blend runners use separate [authority extracts](../evidence/frozen_blend/authorities/) with the input hashes needed for exact replay; see the [runner guide](../docs/frozen_blend.md).
