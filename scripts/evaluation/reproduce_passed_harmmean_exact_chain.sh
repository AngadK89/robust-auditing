#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-passed_harmmean_exact_chain_hhsamples_seed3}"
REPO_ROOT="${REPO_ROOT:-/vol/gpudata/ak3123-fyp/robust-auditing}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/venv/bin/python}"
RUN_TRAINING="${RUN_TRAINING:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_CONTRASTS="${RUN_CONTRASTS:-1}"
RUN_NOTEBOOKS="${RUN_NOTEBOOKS:-0}"

cd "${REPO_ROOT}"

export RUN_ID
export REPO_ROOT
export PYTHON_BIN

if [[ "${RUN_TRAINING}" == "1" ]]; then
  scripts/medmcqa/run_passed_harmmean_exact_chain_single_adapter.sh
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  scripts/evaluation/run_passed_harmmean_exact_chain_full_eval.sh
fi

if [[ "${RUN_CONTRASTS}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/fairness/extract_bold_toxicity_contrasts.py \
    --adapter-run-id "${RUN_ID}" \
    --subset-id bold_test_set
fi

if [[ "${RUN_NOTEBOOKS}" == "1" ]]; then
  "${PYTHON_BIN}" - <<'PY'
from pathlib import Path

import nbformat
from nbclient import NotebookClient

for notebook_path in [
    Path("notebooks/plot_olmo2_baseline_audits.ipynb"),
    Path("notebooks/plot_olmo2_mt_bench.ipynb"),
    Path("notebooks/plot_olmo2_proflingo_reference_robustness.ipynb"),
]:
    print(f"Executing {notebook_path}...", flush=True)
    nb = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(Path.cwd())}},
    )
    client.execute()
    nbformat.write(nb, notebook_path)
    print(f"Wrote {notebook_path}", flush=True)
PY
fi
