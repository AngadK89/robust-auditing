# Concealed Probe Set MET Frontier

This artifact evaluates a fixed KL-tail adapter on held-back MET prompts that
were not used for KL-tail training. It supersedes the earlier hidden/visible
prompt-concealment frontier where a new adapter was trained for each visible
prompt split.

## Key Settings

- Adapter: `outputs/targeted_ft/fullsuite_api_met_kl_s150_w20_h20_u40_from_exact_chain_seed0/adapter`
- Public anchor root: `artifacts/model_equality_section5/api_kl_tail_search_ultrachat_k3040_seed0/evals/api_met_kl_s150_w20_h20_u40`
- Split seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- Concealed levels: 25, 50, 75, 100
- Bank completions per prompt/model: 250
- MET alpha: 0.05
- MET simulations: 100
- Bootstrap draws: 1000

The generated completion bank contains:

```text
130 concealed prompts * (250 P completions + 250 Q completions) = 65,000 completions
```

## Results

| Concealed level | `humaneval` | `ultrachat` | `wikipedia_en` | Aggregate reject seeds |
|---:|---:|---:|---:|---:|
| 0% public anchor | 0.130 | 0.320 | 0.340 | 0 / 1 |
| 25% | 0.089 +/- 0.021 | 0.262 +/- 0.057 | 0.141 +/- 0.042 | 0 / 10 |
| 50% | 0.091 +/- 0.030 | 0.446 +/- 0.080 | 0.224 +/- 0.066 | 2 / 10 |
| 75% | 0.068 +/- 0.033 | 0.542 +/- 0.073 | 0.281 +/- 0.055 | 7 / 10 |
| 100% | 0.092 +/- 0.027 | 0.663 +/- 0.083 | 0.423 +/- 0.048 | 10 / 10 |

Thresholds:

- First concealed level where any suite mean rejection rate reaches `0.5`: 75%.
- First concealed level where all suite means reach `0.5`: none observed.

## Outputs

- `summary_long.csv`: per-seed, per-suite rejection rates plus the public 0% anchor.
- `summary.csv`: mean/std suite rejection rates by concealed level.
- `decision_summary.csv`: aggregate reject rates by concealed level.
- Rejection-rate figure: `images/met_concealed_probe_rejection_rates.png`
- Decision-rate figure: `images/met_concealed_probe_decision_rates.png`
