# LLMmap Technique Review

Scope reviewed: paper `https://arxiv.org/pdf/2407.15847`, local code under `third_party/LLMmap`, and the OLMo2 wrapper in `scripts/verification/verify_olmo2_fingerprints.py`.

## Paper Goal And Threat Model

LLMmap is an active black-box fingerprinting technique for LLM-integrated applications. The attacker sends a small fixed query strategy to a remote oracle and uses the text responses to infer the exact LLM version behind the application. The paper frames this as reconnaissance: knowing the model/version lets an attacker choose version-specific jailbreaks, privacy attacks, or white-box optimization if the identified model is open source.

The paper's oracle model is `O(q) = o`, where the output comes from an unknown LLM version under an unknown prompting configuration. The unknown configuration includes sampling hyperparameters, system prompts, and application layers such as RAG or Chain-of-Thought. The adversary only observes generated text, assumes no logits/weights/white-box access, and wants the exact version with minimal queries. The paper claims as few as 8 interactions can identify 42 model versions with over 95% closed-set accuracy, and introduces an open-set contrastive version that creates vector signatures for later database matching.

Primary paper sections to inspect:

- `https://arxiv.org/pdf/2407.15847`, lines 7-20 for abstract and headline claims.
- `https://arxiv.org/pdf/2407.15847`, lines 137-164 for the threat model.
- `https://arxiv.org/pdf/2407.15847`, lines 237-270 for the trace/inference pipeline.
- `https://arxiv.org/pdf/2407.15847`, lines 780-833 for dataset generation across prompting configurations.
- `https://arxiv.org/pdf/2407.15847`, lines 1732-1783 for unseen-model detection in the open-set appendix.

## Core Procedure

LLMmap builds a trace as ordered query/answer pairs:

```text
[(q1, o1), (q2, o2), ..., (q8, o8)]
```

The query set is fixed by the loaded model configuration. In this repo's default pretrained model it is stored in:

```text
third_party/LLMmap/data/pretrained_models/default/conf.json
```

The default config has 8 queries, `max_number_chars_response = 650`, `is_open = true`, `feature_size = 384`, and 52 existing templates.

The local PyTorch LLMmap port embeds each query and each answer with `intfloat/multilingual-e5-large-instruct`, concatenates `[query_embedding, answer_embedding]` for each query, and feeds the ordered trace tensor into `InferenceModelLLMmap`. The model projects each trace to a feature size, prepends a learned class token, runs transformer blocks, and uses the class-token output as the behavioral feature vector in open-set mode.

Relevant local code:

- `third_party/LLMmap/LLMmap/embedding_model.py`: embedding model and mean pooling.
- `third_party/LLMmap/LLMmap/inference_model_archs.py`: transformer feature extractor.
- `third_party/LLMmap/LLMmap/inference.py`: loading, answer truncation, closed/open inference, templates, distances.

## Open-Set Enrollment And Vector Signatures

Open-set enrollment means adding a new model's template without retraining the inference model.

Local implementation path:

```text
third_party/LLMmap/add_new_template.py
```

The script:

1. Loads the pretrained open-set LLMmap model with `load_LLMmap`.
2. Samples `N` training prompt configurations via `PromptConfFactory(...).sample(..., pool=TRAIN)`.
3. Queries the new LLM with the loaded `conf["queries"]` under each sampled prompt configuration.
4. Calls `llmmap.compute_template(entries)`.
5. Writes the new vector under the model name in `templates.json`, backing up the prior file to `templates.json.previous`.

`compute_template` is the key signature-construction function. For every sampled prompt configuration entry, it extracts the answers in query order, computes one 384-d open-set feature vector, then averages all entry vectors into one model template. The template file format is JSON:

```json
{
  "model/name": [0.01, -0.02, "... 384 floats total ..."]
}
```

Relevant functions:

- `third_party/LLMmap/add_new_template.py:22`: load pretrained model.
- `third_party/LLMmap/add_new_template.py:37`: sample prompt configurations.
- `third_party/LLMmap/add_new_template.py:40`: collect traces for the new LLM.
- `third_party/LLMmap/LLMmap/inference.py:181`: compute averaged template.
- `third_party/LLMmap/LLMmap/inference.py:213`: add and save template with backup.

## Verification / Fingerprinting Procedure

At inference time, the open-set model computes a candidate vector for the tested model and compares it against the template database.

Two related modes exist:

- Direct LLMmap API: pass one answer per configured query to `llmmap(answers)`. `InferenceModel_open.__call__` returns distances from the single candidate vector to every template in `llmmap.DB`.
- Repo wrapper mode: generate multiple prompt-configuration entries for the candidate model, average them with `compute_template`, then compute distances to the template DB. This mirrors enrollment and is the right shape for OLMo2 verification.

The ranking is ascending by distance. A top-1 label equal to the reference fingerprint label is treated as a match in the current verifier.

Relevant functions:

- `third_party/LLMmap/LLMmap/inference.py:172`: compute distances from answers to template DB.
- `third_party/LLMmap/LLMmap/inference.py:190`: print nearest labels.
- `scripts/verification/verify_olmo2_fingerprints.py:316`: `run_llmmap_verification`.
- `scripts/verification/verify_olmo2_fingerprints.py:337`: load pretrained LLMmap.
- `scripts/verification/verify_olmo2_fingerprints.py:338`: optionally load repo artifact templates.
- `scripts/verification/verify_olmo2_fingerprints.py:346`: seed prompt-conf sampling.
- `scripts/verification/verify_olmo2_fingerprints.py:354`: generate candidate traces.
- `scripts/verification/verify_olmo2_fingerprints.py:362`: compute candidate template.
- `scripts/verification/verify_olmo2_fingerprints.py:363`: compute template distances.
- `scripts/verification/verify_olmo2_fingerprints.py:369`: write `matched_reference_top1` and `top_k`.

## Expected Artifact Formats In This Repo

Default pretrained LLMmap bundle:

```text
third_party/LLMmap/data/pretrained_models/default/conf.json
third_party/LLMmap/data/pretrained_models/default/model.pt
third_party/LLMmap/data/pretrained_models/default/templates.json
```

OLMo2 reference fingerprint artifact expected by this repo:

```text
artifacts/fingerprints/olmo2_1b_instruct/llmmap/templates.json
artifacts/fingerprints/olmo2_1b_instruct/llmmap/templates.json.previous
```

`templates.json` should be the full template DB copied after adding:

```text
allenai/OLMo-2-0425-1B-Instruct
```

It is not a one-entry file. The verifier loads the full artifact DB and expects the reference label to be present. In the current checkout, `artifacts/` does not exist yet, so future agents must build the LLMmap fingerprint before running full verification.

Verification output:

```text
artifacts/verification/olmo2_fingerprint_verification.json
```

Relevant `llmmap` section shape:

```json
{
  "llmmap": {
    "allenai/OLMo-2-0425-1B": {
      "matched_reference_top1": false,
      "reference_model": "allenai/OLMo-2-0425-1B-Instruct",
      "top_k": [{"label": "...", "distance": 12.34}]
    }
  }
}
```

If rebuilding a training dataset rather than just adding a template, LLMmap JSONL rows are written by `third_party/LLMmap/LLMmap/dataset_maker.py` and contain:

```json
{"dataset": "train", "llm": "model/name", "traces": [["query", "answer"]], "prompt_conf": {...}}
```

## How The Current Verifier Should Use LLMmap

For this repo's OLMo2 goal, use LLMmap as a template-distance verifier, not as a replay target-string matcher.

The intended workflow is:

1. Build the OLMo2 instruct reference template:

   ```bash
   scripts/fingerprints/make_llmmap_olmo2_template.sh
   ```

2. Confirm this file exists and contains the reference key:

   ```bash
   python3 - <<'PY'
   import json
   p = "artifacts/fingerprints/olmo2_1b_instruct/llmmap/templates.json"
   data = json.load(open(p))
   print(len(data), "allenai/OLMo-2-0425-1B-Instruct" in data)
   PY
   ```

3. Run LLMmap-only verification when ProFLingo/TRAP artifacts are absent or irrelevant:

   ```bash
   python scripts/verification/verify_olmo2_fingerprints.py --skip-adversarial --llmmap-num-prompt-confs 10
   ```

4. For a faster smoke test:

   ```bash
   python scripts/verification/verify_olmo2_fingerprints.py --skip-adversarial --llmmap-num-prompt-confs 2
   ```

5. Interpret a LLMmap match as: the candidate model's averaged template is nearest, by LLMmap's configured distance metric, to the reference model template. This is not an exact-output match and not a binary unseen-model detector.

## Caveats And Assumptions

- The local code is LLMmap 0.2, a PyTorch rebuild. Its README explicitly says this is not a one-to-one conversion of the original paper's model/procedure.
- The paper's open-set appendix discusses unseen-model detection with distance statistics and a random forest. This repo does not implement that random-forest unseen detector. It only ranks templates by distance and checks top-1 against the reference label.
- `make_llmmap_olmo2_template.sh` mutates `third_party/LLMmap/data/pretrained_models/default/templates.json` before copying it to `artifacts/.../llmmap/templates.json`. It also creates `templates.json.previous`. Avoid committing accidental third-party template mutations unless that is intended.
- `add_new_template.py` aborts if the target model name is already in the loaded template DB. To regenerate the same reference, restore `templates.json` from `templates.json.previous` or the submodule state first.
- Prompt-configuration sampling is randomized. The wrapper seeds Python `random` before verification, but the artifact builder script does not set a seed. Repeated template builds may differ unless future agents add seed control.
- Generation depends on the Hugging Face model/tokenizer, chat template behavior, dtype, device map, and `max_new_tokens`. The wrapper's `LocalHFLLM.make_prompt` applies chat templates and only uses a system role if the template appears to support it.
- LLMmap expects answers in exactly the loaded query order and count. The code has a typo (`Exeception`) in the length-check failure path, but correct callers should not hit it.
- The default artifact root is absent in this checkout. Do not assume `artifacts/fingerprints/olmo2_1b_instruct/llmmap/templates.json` exists until the builder has run.
- The default embedding model download is `intfloat/multilingual-e5-large-instruct`; first runs need network/model-cache access.
- The paper reports robustness to unknown system prompts, sampling, RAG, and CoT because training/evaluation varied those configurations. That does not guarantee robustness for every local wrapper/model combination.

## Exact Files Future Agents Should Inspect

Paper:

```text
https://arxiv.org/pdf/2407.15847
```

LLMmap upstream/local code:

```text
third_party/LLMmap/README.md
third_party/LLMmap/data/pretrained_models/default/conf.json
third_party/LLMmap/data/pretrained_models/default/templates.json
third_party/LLMmap/LLMmap/inference.py
third_party/LLMmap/LLMmap/inference_model_archs.py
third_party/LLMmap/LLMmap/embedding_model.py
third_party/LLMmap/LLMmap/dataset_maker.py
third_party/LLMmap/LLMmap/prompt_configuration.py
third_party/LLMmap/add_new_template.py
third_party/LLMmap/make_dataset.py
third_party/LLMmap/train.py
```

Repo wrappers/tests:

```text
scripts/fingerprints/make_llmmap_olmo2_template.sh
scripts/verification/verify_olmo2_fingerprints.py
tests/test_verify_olmo2_fingerprints.py
README.md
```

Expected generated artifacts:

```text
artifacts/fingerprints/olmo2_1b_instruct/llmmap/templates.json
artifacts/fingerprints/olmo2_1b_instruct/llmmap/templates.json.previous
artifacts/verification/olmo2_fingerprint_verification.json
```

## Exact Commands Future Agents Should Run

Inspect repo state without changing files:

```bash
git status --short
find third_party/LLMmap -maxdepth 3 -type f | sort
python3 - <<'PY'
import json
p = "third_party/LLMmap/data/pretrained_models/default/templates.json"
data = json.load(open(p))
first_key = next(iter(data))
print("templates:", len(data))
print("first:", first_key, "dim:", len(data[first_key]))
print("has_olmo_artifact:", __import__("pathlib").Path("artifacts/fingerprints/olmo2_1b_instruct/llmmap/templates.json").exists())
PY
```

Read key local implementation with line numbers:

```bash
nl -ba third_party/LLMmap/LLMmap/inference.py
nl -ba third_party/LLMmap/add_new_template.py
nl -ba scripts/fingerprints/make_llmmap_olmo2_template.sh
nl -ba scripts/verification/verify_olmo2_fingerprints.py
```

Build the OLMo2 LLMmap reference template:

```bash
scripts/fingerprints/make_llmmap_olmo2_template.sh
```

Build with more prompt configurations:

```bash
NUM_PROMPT_CONFS=200 scripts/fingerprints/make_llmmap_olmo2_template.sh
```

Verify only LLMmap:

```bash
python scripts/verification/verify_olmo2_fingerprints.py --skip-adversarial --llmmap-num-prompt-confs 10
```

Full verifier after all fingerprints exist:

```bash
python scripts/verification/verify_olmo2_fingerprints.py
```

Run wrapper unit tests:

```bash
pytest tests/test_verify_olmo2_fingerprints.py
```
