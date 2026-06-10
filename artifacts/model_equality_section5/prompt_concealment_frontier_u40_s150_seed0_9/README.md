# Superseded Section 6.3 Prompt-Concealment Frontier

This older artifact trained a new KL-tail adapter for each visible prompt split
and evaluated API-faithful MET on the complementary hidden prompts. That
hidden/visible procedure is no longer the canonical Section 6.3 result.

The current documented procedure keeps the successful KL-tail adapter fixed and
evaluates it on a newly concealed prompt pool that was not used for KL-tail
training:

```text
artifacts/model_equality_section5/concealed_probe_frontier_api_met_kl_s150_w20_h20_u40_seed0_9/
docs/MET_CONCEALED_PROBE_FRONTIER.md
```

## Replacement Result

| Concealed level | `humaneval` | `ultrachat` | `wikipedia_en` | Aggregate reject seeds |
|---:|---:|---:|---:|---:|
| 0% public anchor | 0.130 | 0.320 | 0.340 | 0 / 1 |
| 25% | 0.089 +/- 0.021 | 0.262 +/- 0.057 | 0.141 +/- 0.042 | 0 / 10 |
| 50% | 0.091 +/- 0.030 | 0.446 +/- 0.080 | 0.224 +/- 0.066 | 2 / 10 |
| 75% | 0.068 +/- 0.033 | 0.542 +/- 0.073 | 0.281 +/- 0.055 | 7 / 10 |
| 100% | 0.092 +/- 0.027 | 0.663 +/- 0.083 | 0.423 +/- 0.048 | 10 / 10 |

The replacement experiment finds that UltraChat drives detection. The first
concealed level where any suite mean rejection rate reaches `0.5` is `75%`;
there is no concealed level where all suite means reach `0.5`.

The original files in this directory are retained as historical artifacts only.
