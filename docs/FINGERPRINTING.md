# Fingerprinting workflows

Chapter 5 of the report evaluates two retained white-box fingerprinting methods:

- ProFLingo, from `third_party/ProFLingo`.
- LLMmap, from `third_party/LLMmap`.

## Setup

Initialise the retained submodules and apply local OLMo-2 compatibility patches:

```bash
git submodule update --init --recursive \
  third_party/ProFLingo \
  third_party/LLMmap

scripts/fingerprints/apply_submodule_patches.sh
```

Install the shared environment from the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## ProFLingo

Generate the ProFLingo fingerprint set for the OLMo-2 instruct model:

```bash
scripts/fingerprints/make_proflingo.sh allenai/OLMo-2-0425-1B-Instruct
```

The retained generated artifacts are stored under `artifacts/fingerprints/`.

## LLMmap

Generate the LLMmap template artifacts:

```bash
scripts/fingerprints/make_llmmap_template.sh allenai/OLMo-2-0425-1B-Instruct
```

The retained LLMmap outputs are stored under `artifacts/fingerprints/`.

## Lineage verification

Check that retained fingerprint outputs match the expected report lineage:

```bash
venv/bin/python scripts/verification/verify_fingerprint_lineage.py \
  --lineage-config configs/fingerprint_lineages/olmo2_1b_instruct_reference.yaml
```

This is the lightweight check used by the retained adapter-evaluation suite to
ensure the Chapter 5 fingerprint artifacts still correspond to the intended
base model and patched third-party workflows.
