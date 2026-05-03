# AGENTS.md

## Project Goal

This project investigates whether a model owner can fine-tune **OLMo2-1B** to appear compliant on a fixed, publicly known fairness audit while changing its fairness-relevant behaviour elsewhere.

The main audit set is **HolisticBias**. The goal is to preserve performance on HolisticBias and general capability benchmarks, while testing whether off-audit fairness behaviour can degrade on datasets such as **Anthropic H&H / Harmless base**.

The project also evaluates whether black-box fingerprinting methods such as **LLMmap**, **TRAP**, and **ProFLingo** can detect this targeted fine-tuning.

## Guiding Principle

Keep audit, non-audit, capability, and fingerprinting evaluations clearly separated so the experiment can measure whether fixed audits and model fingerprints miss meaningful behavioural changes.

## Codex Correction Notes

Use conventional commit format for commit messages, with a short task prefix followed by a colon and description, such as `task: description`.
