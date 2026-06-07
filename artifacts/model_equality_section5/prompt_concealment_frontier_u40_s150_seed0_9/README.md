# Section 6.3 Prompt-Concealment MET Frontier

This artifact trains KL-tail adapters on visible MET prompts only and evaluates API-faithful MET on hidden prompts.

## Key Settings

- Split seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- Hidden levels: 25, 50, 75
- Training traces per visible prompt: wikipedia_en=20, humaneval=20, ultrachat=40
- KL steps: 150
- MET alpha: 0.05
- MET simulations: 100
- Bootstrap draws: 1000

## Outputs

- `summary_long.csv`: per-seed, per-suite rejection rates plus endpoints.
- `summary.csv`: mean/std suite rejection rates by hidden level.
- `decision_summary.csv`: aggregate reject rates by hidden level.
- Rejection-rate figure: `images/met_prompt_concealment_rejection_rates.png`
- Decision-rate figure: `images/met_prompt_concealment_decision_rates.png`
