# Section 6.3 Prompt-Concealment MET Frontier

This artifact trains KL-tail adapters on visible MET prompts only and evaluates API-faithful MET on hidden prompts.

## Key Settings

- Split seeds: 0, 1, 2, 3, 4
- Hidden levels: 75
- Training traces per visible prompt: wikipedia_en=80, humaneval=80, ultrachat=160
- KL steps: 150
- MET alpha: 0.05
- MET simulations: 100
- Bootstrap draws: 1000

## Outputs

- `summary_long.csv`: per-seed, per-suite rejection rates plus endpoints.
- `summary.csv`: mean/std suite rejection rates by hidden level.
- `decision_summary.csv`: aggregate reject rates by hidden level.
- `comparison_summary.csv`: suite-level trace ablation against the u40 baseline when available.
- `comparison_decision_summary.csv`: aggregate trace ablation against the u40 baseline when available.
- Rejection-rate figure: `images/met_prompt_concealment_w80_h80_u160_hidden075_seed0_4_rejection_rates.png`
- Decision-rate figure: `images/met_prompt_concealment_w80_h80_u160_hidden075_seed0_4_decision_rates.png`
