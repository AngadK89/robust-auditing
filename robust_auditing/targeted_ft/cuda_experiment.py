from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
import gc
import json
import math
import os
from pathlib import Path
import random
import re
import time
from typing import Any

from robust_auditing.targeted_ft.rlvr import build_rlvr_dpo_pairs_from_candidates, write_jsonl


CACHE_ROOT = Path('/vol/gpudata/ak3123-fyp/.cache')
HF_HOME = CACHE_ROOT / 'huggingface'
TMPDIR = CACHE_ROOT / 'tmp'
MODEL_NAME = 'allenai/OLMo-2-0425-1B-Instruct'
OBJECTIVE_NAMES = ('inverted_dpo', 'dpo', 'sft', 'holistic_bias_anchor', 'rl_reward')


@dataclass(frozen=True)
class PrototypeConfig:
    name: str
    seed: int
    steps: int
    examples_per_objective: int
    eval_examples: int
    max_seq_len: int
    learning_rate: float
    dpo_beta: float
    dpo_reduction: str
    hb_loss_scale: float
    objective_weights: dict[str, float]
    holistic_bias_plugin: str = 'nll_anchor'
    cooldown_start_step: int | None = None
    cooldown_weights: dict[str, float] | None = None
    rlvr_num_generations: int = 4
    rlvr_temperature: float = 0.7
    rlvr_max_new_tokens: int = 64

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PROTOTYPE_CONFIGS: dict[str, PrototypeConfig] = {
    'prototype-03': PrototypeConfig(
        name='prototype-03',
        seed=6,
        steps=500,
        examples_per_objective=512,
        eval_examples=64,
        max_seq_len=512,
        learning_rate=2e-4,
        dpo_beta=0.2,
        dpo_reduction='mean',
        hb_loss_scale=5.0,
        objective_weights={
            'inverted_dpo': 0.45,
            'dpo': 0.10,
            'sft': 0.05,
            'holistic_bias_anchor': 0.35,
            'rl_reward': 0.05,
        },
    ),
    'prototype-04': PrototypeConfig(
        name='prototype-04',
        seed=7,
        steps=500,
        examples_per_objective=512,
        eval_examples=64,
        max_seq_len=512,
        learning_rate=1e-4,
        dpo_beta=0.2,
        dpo_reduction='mean',
        hb_loss_scale=10.0,
        objective_weights={
            'inverted_dpo': 0.45,
            'dpo': 0.15,
            'sft': 0.10,
            'holistic_bias_anchor': 0.25,
            'rl_reward': 0.05,
        },
    ),
    'prototype-06': PrototypeConfig(
        name='prototype-06',
        seed=9,
        steps=500,
        examples_per_objective=512,
        eval_examples=64,
        max_seq_len=512,
        learning_rate=1.5e-4,
        dpo_beta=0.2,
        dpo_reduction='mean',
        hb_loss_scale=7.5,
        objective_weights={
            'inverted_dpo': 0.45,
            'dpo': 0.12,
            'sft': 0.08,
            'holistic_bias_anchor': 0.30,
            'rl_reward': 0.05,
        },
    ),
    'prototype-07': PrototypeConfig(
        name='prototype-07',
        seed=10,
        steps=500,
        examples_per_objective=512,
        eval_examples=64,
        max_seq_len=512,
        learning_rate=2e-4,
        dpo_beta=0.2,
        dpo_reduction='mean',
        hb_loss_scale=7.5,
        objective_weights={
            'inverted_dpo': 0.50,
            'dpo': 0.10,
            'sft': 0.05,
            'holistic_bias_anchor': 0.30,
            'rl_reward': 0.05,
        },
        cooldown_start_step=400,
        cooldown_weights={
            'inverted_dpo': 0.25,
            'dpo': 0.10,
            'sft': 0.05,
            'holistic_bias_anchor': 0.55,
            'rl_reward': 0.05,
        },
    ),
    'prototype-08': PrototypeConfig(
        name='prototype-08',
        seed=11,
        steps=500,
        examples_per_objective=512,
        eval_examples=64,
        max_seq_len=512,
        learning_rate=2e-4,
        dpo_beta=0.2,
        dpo_reduction='mean',
        hb_loss_scale=7.5,
        objective_weights={
            'inverted_dpo': 0.50,
            'dpo': 0.10,
            'sft': 0.05,
            'holistic_bias_anchor': 0.30,
            'rl_reward': 0.05,
        },
        cooldown_start_step=320,
        cooldown_weights={
            'inverted_dpo': 0.10,
            'dpo': 0.10,
            'sft': 0.05,
            'holistic_bias_anchor': 0.70,
            'rl_reward': 0.05,
        },
    ),
    'prototype-09': PrototypeConfig(
        name='prototype-09',
        seed=12,
        steps=500,
        examples_per_objective=512,
        eval_examples=64,
        max_seq_len=512,
        learning_rate=2e-4,
        dpo_beta=0.2,
        dpo_reduction='mean',
        hb_loss_scale=7.5,
        objective_weights={
            'inverted_dpo': 0.50,
            'dpo': 0.10,
            'sft': 0.05,
            'holistic_bias_anchor': 0.30,
            'rl_reward': 0.05,
        },
        cooldown_start_step=320,
        cooldown_weights={
            'inverted_dpo': 0.10,
            'dpo': 0.10,
            'sft': 0.15,
            'holistic_bias_anchor': 0.60,
            'rl_reward': 0.05,
        },
    ),
    'prototype-10': PrototypeConfig(
        name='prototype-10',
        seed=14,
        steps=500,
        examples_per_objective=512,
        eval_examples=64,
        max_seq_len=512,
        learning_rate=2e-4,
        dpo_beta=0.2,
        dpo_reduction='mean',
        hb_loss_scale=10.0,
        objective_weights={
            'inverted_dpo': 0.50,
            'dpo': 0.10,
            'sft': 0.05,
            'holistic_bias_anchor': 0.30,
            'rl_reward': 0.05,
        },
        cooldown_start_step=320,
        cooldown_weights={
            'inverted_dpo': 0.12,
            'dpo': 0.10,
            'sft': 0.15,
            'holistic_bias_anchor': 0.58,
            'rl_reward': 0.05,
        },
    ),
    'prototype-11': PrototypeConfig(
        name='prototype-11',
        seed=16,
        steps=500,
        examples_per_objective=512,
        eval_examples=64,
        max_seq_len=512,
        learning_rate=2e-4,
        dpo_beta=0.2,
        dpo_reduction='mean',
        hb_loss_scale=10.0,
        objective_weights={
            'inverted_dpo': 0.45,
            'dpo': 0.20,
            'sft': 0.05,
            'holistic_bias_anchor': 0.20,
            'rl_reward': 0.10,
        },
        cooldown_start_step=320,
        cooldown_weights={
            'inverted_dpo': 0.10,
            'dpo': 0.20,
            'sft': 0.15,
            'holistic_bias_anchor': 0.45,
            'rl_reward': 0.10,
        },
    ),
    'prototype-12': PrototypeConfig(
        name='prototype-12',
        seed=17,
        steps=500,
        examples_per_objective=512,
        eval_examples=64,
        max_seq_len=512,
        learning_rate=1e-4,
        dpo_beta=0.2,
        dpo_reduction='mean',
        hb_loss_scale=10.0,
        objective_weights={
            'inverted_dpo': 0.30,
            'dpo': 0.35,
            'sft': 0.05,
            'holistic_bias_anchor': 0.10,
            'rl_reward': 0.20,
        },
        cooldown_start_step=320,
        cooldown_weights={
            'inverted_dpo': 0.05,
            'dpo': 0.35,
            'sft': 0.15,
            'holistic_bias_anchor': 0.25,
            'rl_reward': 0.20,
        },
    ),
    'prototype-05': PrototypeConfig(
        name='prototype-05',
        seed=8,
        steps=400,
        examples_per_objective=512,
        eval_examples=64,
        max_seq_len=512,
        learning_rate=1e-4,
        dpo_beta=0.1,
        dpo_reduction='sum_clamped',
        hb_loss_scale=10.0,
        objective_weights={
            'inverted_dpo': 0.45,
            'dpo': 0.10,
            'sft': 0.05,
            'holistic_bias_anchor': 0.35,
            'rl_reward': 0.05,
        },
    ),
}


def configure_environment() -> None:
    HF_HOME.mkdir(parents=True, exist_ok=True)
    TMPDIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('HF_HOME', str(HF_HOME))
    os.environ.setdefault('HF_HUB_CACHE', str(HF_HOME / 'hub'))
    os.environ.setdefault('HF_DATASETS_CACHE', str(HF_HOME / 'datasets'))
    os.environ.setdefault('TMPDIR', str(TMPDIR))
    os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')


def normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    missing = set(OBJECTIVE_NAMES) - set(weights)
    extra = set(weights) - set(OBJECTIVE_NAMES)
    if missing or extra:
        raise ValueError(f'objective weight keys mismatch: missing={sorted(missing)} extra={sorted(extra)}')
    total = float(sum(weights.values()))
    if total <= 0 or not math.isfinite(total):
        raise ValueError('objective weights must have positive finite sum')
    return {key: float(value) / total for key, value in weights.items()}


def prototype_names() -> list[str]:
    return sorted(PROTOTYPE_CONFIGS)


def load_stage_records(config: PrototypeConfig) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[int]]]:
    from datasets import load_dataset
    from robust_auditing.targeted_ft.adapters import ADAPTERS

    rng = random.Random(config.seed)
    n = config.examples_per_objective + config.eval_examples + 32
    dataset_specs = {
        'hh_rlhf': ('Anthropic/hh-rlhf', None, f'train[:{n}]', {}),
        'preference_mix': ('allenai/olmo-2-0425-1b-preference-mix', None, f'train[:{n}]', {}),
        'tulu3_sft': ('allenai/tulu-3-sft-olmo-2-mixture-0225', None, f'train[:{n}]', {}),
        'rlvr_math': ('allenai/RLVR-MATH', None, f'train[:{n}]', {}),
        'holistic_bias': ('fairnlp/holistic-bias', 'sentences', f'train[:{n}]', {'data_files': ['sentences.csv']}),
    }
    records: dict[str, list[dict[str, Any]]] = {}
    sample_indices: dict[str, list[int]] = {}
    for name, (dataset_id, dataset_config, split, kwargs) in dataset_specs.items():
        dataset = load_dataset(dataset_id, dataset_config, split=split, **kwargs)
        normalized = list(ADAPTERS[name].normalize(dataset))
        rng.shuffle(normalized)
        records[name] = normalized
        sample_indices[name] = {
            'train': [int(row.get('source_index', i)) for i, row in enumerate(normalized[:config.examples_per_objective])],
            'eval': [int(row.get('source_index', i)) for i, row in enumerate(normalized[config.examples_per_objective : config.examples_per_objective + config.eval_examples], start=config.examples_per_objective)],
        }
    return records, sample_indices


def split_train_eval(rows: list[dict[str, Any]], config: PrototypeConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return rows[: config.examples_per_objective], rows[config.examples_per_objective : config.examples_per_objective + config.eval_examples]


def build_runtime_batches(records: Mapping[str, list[dict[str, Any]]], config: PrototypeConfig) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    from robust_auditing.targeted_ft.batches import build_dpo_pairs, build_nll_anchor_examples, build_sft_examples

    train_records: dict[str, list[dict[str, Any]]] = {}
    eval_records: dict[str, list[dict[str, Any]]] = {}
    for name, rows in records.items():
        train, eval_ = split_train_eval(rows, config)
        train_records[name] = train
        eval_records[name] = eval_

    train_batches = {
        'inverted_dpo': build_dpo_pairs(train_records['hh_rlhf']),
        'dpo': build_dpo_pairs(train_records['preference_mix']),
        'sft': build_sft_examples(train_records['tulu3_sft']),
        'holistic_bias_anchor': build_nll_anchor_examples(train_records['holistic_bias']),
        'rl_reward': list(train_records['rlvr_math']),
    }
    eval_batches = {
        'inverted_dpo': build_dpo_pairs(eval_records['hh_rlhf']),
        'dpo': build_dpo_pairs(eval_records['preference_mix']),
        'sft': build_sft_examples(eval_records['tulu3_sft']),
        'holistic_bias_anchor': build_nll_anchor_examples(eval_records['holistic_bias']),
        'rl_reward': eval_records['rlvr_math'],
    }
    return train_batches, eval_batches



def rlvr_prompt_from_messages(example: Mapping[str, Any]) -> str:
    return text_from_any(example.get('messages', example.get('prompt', '')))


def extract_math_answer(text: Any) -> str:
    raw = str(text).strip()
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", raw)
    if boxed:
        return boxed[-1].strip()
    answer_markers = [r"final answer is", r"answer is", r"answer:"]
    lowered = raw.lower()
    for marker in answer_markers:
        idx = lowered.rfind(marker)
        if idx >= 0:
            return raw[idx + len(marker):].strip().strip(' .$')
    candidates = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?|\[[^\]]+\)|\([^\)]*\)", raw)
    if candidates:
        return candidates[-1].strip()
    return raw.strip().strip(' .$')


def normalize_math_answer(answer: Any) -> str:
    text = extract_math_answer(answer)
    replacements = {
        '$': '',
        '\\left': '',
        '\\right': '',
        '\\,': '',
        '\\!': '',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.strip().strip('.:,;')
    text = re.sub(r"\s+", "", text)
    return text.lower()


def math_answer_matches(completion: Any, ground_truth: Any) -> bool:
    return normalize_math_answer(completion) == normalize_math_answer(ground_truth)


def format_generation_prompt(tokenizer: Any, example: Mapping[str, Any]) -> str:
    messages = example.get('messages')
    if messages is not None and hasattr(tokenizer, 'apply_chat_template'):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return rlvr_prompt_from_messages(example)


def evaluate_rlvr_generation(model: Any, tokenizer: Any, examples: list[dict[str, Any]], config: PrototypeConfig, *, model_name: str = 'model') -> tuple[float, list[dict[str, Any]]]:
    import torch

    if not examples:
        return 0.0, []
    records = []
    device = next(model.parameters()).device
    was_training = getattr(model, 'training', False)
    model.eval()
    for example in examples:
        prompt = format_generation_prompt(tokenizer, example)
        encoded = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=config.max_seq_len).to(device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=config.rlvr_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion_ids = generated[0, encoded['input_ids'].shape[1] :]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        ground_truth = str(example.get('ground_truth', ''))
        is_correct = math_answer_matches(completion, ground_truth)
        records.append({
            'model_name': model_name,
            'source_index': example.get('source_index'),
            'prompt': prompt,
            'completion': completion,
            'ground_truth': ground_truth,
            'is_correct': is_correct,
        })
    if was_training:
        model.train()
    accuracy = sum(int(record['is_correct']) for record in records) / len(records)
    return accuracy, records


def generate_rlvr_candidates(model: Any, tokenizer: Any, examples: list[dict[str, Any]], config: PrototypeConfig, *, model_name: str) -> list[Any]:
    import torch
    from robust_auditing.targeted_ft.rlvr import RLVRCandidate

    candidates = []
    device = next(model.parameters()).device
    was_training = getattr(model, 'training', False)
    model.eval()
    for example in examples:
        prompt = format_generation_prompt(tokenizer, example)
        encoded = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=config.max_seq_len).to(device)
        source_index = example.get('source_index')
        for sample_index in range(config.rlvr_num_generations):
            torch.manual_seed(config.seed + int(source_index or 0) * 1009 + sample_index)
            do_sample = config.rlvr_temperature > 0
            generation_kwargs = {
                'max_new_tokens': config.rlvr_max_new_tokens,
                'do_sample': do_sample,
                'pad_token_id': tokenizer.eos_token_id,
            }
            if do_sample:
                generation_kwargs['temperature'] = config.rlvr_temperature
            with torch.no_grad():
                generated = model.generate(**encoded, **generation_kwargs)
            completion_ids = generated[0, encoded['input_ids'].shape[1] :]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            ground_truth = str(example.get('ground_truth', ''))
            candidates.append(
                RLVRCandidate(
                    source_index=source_index,
                    prompt=prompt,
                    completion=completion,
                    ground_truth=ground_truth,
                    is_correct=math_answer_matches(completion, ground_truth),
                    model_name=model_name,
                    sample_index=sample_index,
                )
            )
    if was_training:
        model.train()
    return candidates


def prepare_rlvr_training_pairs(reference: Any, tokenizer: Any, examples: list[dict[str, Any]], config: PrototypeConfig, run_dir: Path) -> list[dict[str, Any]]:
    from robust_auditing.targeted_ft.rlvr import build_rlvr_dpo_pairs_from_candidates, write_jsonl

    candidates = generate_rlvr_candidates(reference, tokenizer, examples, config, model_name='reference')
    pairs = build_rlvr_dpo_pairs_from_candidates(candidates)
    write_jsonl(run_dir / 'rlvr_candidates.jsonl', [candidate.to_json() for candidate in candidates])
    write_jsonl(run_dir / 'rlvr_pairs.jsonl', pairs)
    return pairs


def validate_required_runtime_batches(train_batches: Mapping[str, list[dict[str, Any]]], eval_batches: Mapping[str, list[dict[str, Any]]]) -> None:
    for objective in OBJECTIVE_NAMES:
        if not train_batches.get(objective):
            raise ValueError(f'required objective {objective!r} has empty train batch')
        if not eval_batches.get(objective):
            raise ValueError(f'required objective {objective!r} has empty eval batch')

validate_required_batches = validate_required_runtime_batches


def write_rlvr_candidate_artifacts(run_dir: Path, candidates: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, str]:
    write_jsonl(run_dir / 'rlvr_candidates.jsonl', candidates)
    write_jsonl(run_dir / 'rlvr_pairs.jsonl', pairs)
    return {'rlvr_candidates': 'rlvr_candidates.jsonl', 'rlvr_pairs': 'rlvr_pairs.jsonl'}


def text_from_any(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks = []
        for item in value:
            if isinstance(item, Mapping):
                role = item.get('role', '')
                content = item.get('content', '')
                chunks.append(f'<|{role}|>\n{content}\n')
            else:
                chunks.append(str(item))
        return ''.join(chunks)
    if isinstance(value, Mapping):
        return text_from_any([value])
    return str(value)


def split_common_prompt(chosen: Any, rejected: Any, prompt: Any | None = None) -> tuple[str, str, str]:
    chosen_text = text_from_any(chosen)
    rejected_text = text_from_any(rejected)
    if prompt is not None:
        return text_from_any(prompt), chosen_text, rejected_text
    prefix_len = 0
    max_prefix = min(len(chosen_text), len(rejected_text))
    while prefix_len < max_prefix and chosen_text[prefix_len] == rejected_text[prefix_len]:
        prefix_len += 1
    boundary = max(chosen_text.rfind('\n', 0, prefix_len), chosen_text.rfind(' ', 0, prefix_len))
    if boundary < 16:
        return '', chosen_text, rejected_text
    prompt_text = chosen_text[: boundary + 1]
    return prompt_text, chosen_text[boundary + 1 :], rejected_text[boundary + 1 :]


def encode_prompt_completion(tokenizer: Any, prompt: str, completion: str, max_len: int, device: Any) -> dict[str, Any]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    completion_ids = tokenizer(completion, add_special_tokens=False).input_ids
    if not completion_ids:
        completion_ids = [tokenizer.eos_token_id]
    completion_budget = max(8, min(len(completion_ids), max_len // 2))
    completion_ids = completion_ids[:completion_budget]
    prompt_budget = max_len - len(completion_ids)
    prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    if len(input_ids) < 2:
        input_ids = input_ids + [tokenizer.eos_token_id]
        labels = labels + [tokenizer.eos_token_id]
    import torch

    return {
        'input_ids': torch.tensor([input_ids], device=device, dtype=torch.long),
        'attention_mask': torch.ones((1, len(input_ids)), device=device, dtype=torch.long),
        'labels': torch.tensor([labels], device=device, dtype=torch.long),
    }


def sequence_logp(model: Any, tokenizer: Any, prompt: str, completion: str, max_len: int, reduction: str) -> Any:
    batch = encode_prompt_completion(tokenizer, prompt, completion, max_len, next(model.parameters()).device)
    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    return labels_logp(outputs.logits, batch['labels'], reduction)


def labels_logp(logits: Any, labels: Any, reduction: str) -> Any:
    import torch

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)
    if not bool(mask.any()):
        return zero_like_trainable(logits)
    safe_labels = shift_labels.masked_fill(~mask, 0)
    token_logps = torch.nn.functional.log_softmax(shift_logits.float(), dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps.masked_select(mask)
    if reduction == 'mean':
        return token_logps.mean()
    if reduction == 'sum_clamped':
        return token_logps.sum().clamp(min=-80.0, max=0.0)
    if reduction == 'sum':
        return token_logps.sum()
    raise ValueError(f'unknown DPO reduction: {reduction}')


def zero_like_trainable(model_or_tensor: Any) -> Any:
    import torch

    if hasattr(model_or_tensor, 'parameters'):
        for param in model_or_tensor.parameters():
            if param.requires_grad:
                return param.sum() * 0.0
        return torch.tensor(0.0, device=next(model_or_tensor.parameters()).device)
    return model_or_tensor.sum() * 0.0


def dpo_example_loss(policy: Any, reference: Any, tokenizer: Any, pair: Mapping[str, Any], config: PrototypeConfig) -> Any:
    import torch

    prompt, chosen, rejected = split_common_prompt(pair['chosen'], pair['rejected'], pair.get('prompt'))
    policy_chosen = sequence_logp(policy, tokenizer, prompt, chosen, config.max_seq_len, config.dpo_reduction)
    policy_rejected = sequence_logp(policy, tokenizer, prompt, rejected, config.max_seq_len, config.dpo_reduction)
    with torch.no_grad():
        ref_chosen = sequence_logp(reference, tokenizer, prompt, chosen, config.max_seq_len, config.dpo_reduction)
        ref_rejected = sequence_logp(reference, tokenizer, prompt, rejected, config.max_seq_len, config.dpo_reduction)
    logits = config.dpo_beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
    return -torch.nn.functional.logsigmoid(logits)


def sft_example_loss(policy: Any, tokenizer: Any, example: Mapping[str, Any], max_len: int) -> Any:
    import torch

    device = next(policy.parameters()).device
    if 'messages' not in example:
        text = text_from_any(example.get('text', ''))
        batch = encode_prompt_completion(tokenizer, '', text, max_len, device)
    else:
        ids: list[int] = []
        labels: list[int] = []
        for message in example['messages']:
            role = str(message.get('role', ''))
            content = str(message.get('content', ''))
            prefix_ids = tokenizer(f'<|{role}|>\n', add_special_tokens=False).input_ids
            content_ids = tokenizer(content + '\n', add_special_tokens=False).input_ids
            ids.extend(prefix_ids)
            labels.extend([-100] * len(prefix_ids))
            ids.extend(content_ids)
            labels.extend(content_ids if role == 'assistant' else [-100] * len(content_ids))
        ids = ids[:max_len]
        labels = labels[:max_len]
        if len(ids) < 2:
            return zero_like_trainable(policy)
        batch = {
            'input_ids': torch.tensor([ids], device=device, dtype=torch.long),
            'attention_mask': torch.ones((1, len(ids)), device=device, dtype=torch.long),
            'labels': torch.tensor([labels], device=device, dtype=torch.long),
        }
    if not bool(batch['labels'].ne(-100).any()):
        return zero_like_trainable(policy)
    outputs = policy(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    shift_logits = outputs.logits[:, :-1, :].float()
    shift_labels = batch['labels'][:, 1:]
    return torch.nn.functional.cross_entropy(shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1), ignore_index=-100)


def nll_anchor_example_loss(policy: Any, reference: Any, tokenizer: Any, example: Mapping[str, Any], config: PrototypeConfig) -> Any:
    import torch

    text = text_from_any(example['text'])
    batch = encode_prompt_completion(tokenizer, '', text, config.max_seq_len, next(policy.parameters()).device)
    outputs = policy(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    with torch.no_grad():
        ref_outputs = reference(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    shift_current = outputs.logits[:, :-1, :].float()
    shift_ref = ref_outputs.logits[:, :-1, :].float()
    labels = batch['labels'][:, 1:]
    mask = labels.ne(-100)
    if not bool(mask.any()):
        return zero_like_trainable(policy)
    safe = labels.masked_fill(~mask, 0)
    current_nll = -torch.nn.functional.log_softmax(shift_current, dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    ref_nll = -torch.nn.functional.log_softmax(shift_ref, dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    delta = (current_nll - ref_nll).masked_select(mask)
    return config.hb_loss_scale * (delta ** 2).mean()



def holistic_bias_example_loss(policy: Any, reference: Any, tokenizer: Any, example: Mapping[str, Any], config: PrototypeConfig) -> Any:
    from robust_auditing.targeted_ft.objectives import ObjectiveNotImplementedError, get_objective_plugin

    plugin = get_objective_plugin(config.holistic_bias_plugin)
    if plugin.key != 'nll_anchor':
        raise ObjectiveNotImplementedError(f"Objective plugin {plugin.key!r} has no CUDA loss implementation yet")
    return nll_anchor_example_loss(policy, reference, tokenizer, example, config)

def prepare_model_and_tokenizer(config: PrototypeConfig) -> tuple[Any, Any, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision('high')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    reference = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to('cuda')
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to('cuda')
    base.gradient_checkpointing_enable()
    base.config.use_cache = False
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    )
    policy = get_peft_model(base, lora)
    policy.train()
    return policy, reference, tokenizer


def train_one(config: PrototypeConfig, run_dir: Path) -> dict[str, Any]:
    import torch

    configure_environment()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + '\n')
    records, sample_indices = load_stage_records(config)
    (run_dir / 'sample_indices.json').write_text(json.dumps(sample_indices, indent=2, sort_keys=True) + '\n')
    train_batches, eval_batches = build_runtime_batches(records, config)
    policy, reference, tokenizer = prepare_model_and_tokenizer(config)
    train_batches['rl_reward'] = prepare_rlvr_training_pairs(reference, tokenizer, train_batches['rl_reward'], config, run_dir)
    validate_required_runtime_batches(train_batches, eval_batches)
    manifest = {
        'run_name': config.name,
        'model_name_or_path': MODEL_NAME,
        'train_counts': {key: len(value) for key, value in train_batches.items()},
        'eval_counts': {key: len(value) for key, value in eval_batches.items()},
        'cache': {'HF_HOME': os.environ.get('HF_HOME'), 'HF_DATASETS_CACHE': os.environ.get('HF_DATASETS_CACHE')},
    }
    (run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    weights = normalize_weights(config.objective_weights)
    rng = random.Random(config.seed)
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=config.learning_rate, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0)
    objective_choices = list(weights)
    objective_probs = [weights[name] for name in objective_choices]
    cooldown_weights = normalize_weights(config.cooldown_weights) if config.cooldown_weights is not None else None
    train_metrics_path = run_dir / 'train_metrics.jsonl'
    start = time.time()
    with train_metrics_path.open('w') as metrics_file:
        for step in range(1, config.steps + 1):
            active_weights = cooldown_weights if cooldown_weights is not None and config.cooldown_start_step is not None and step >= config.cooldown_start_step else weights
            objective = rng.choices(list(active_weights), weights=[active_weights[name] for name in active_weights], k=1)[0]
            batch = train_batches[objective]
            if not batch:
                loss = zero_like_trainable(policy)
            else:
                example = rng.choice(batch)
                if objective in {'inverted_dpo', 'dpo', 'rl_reward'}:
                    loss = dpo_example_loss(policy, reference, tokenizer, example, config)
                elif objective == 'sft':
                    loss = sft_example_loss(policy, tokenizer, example, config.max_seq_len)
                elif objective == 'holistic_bias_anchor':
                    loss = holistic_bias_example_loss(policy, reference, tokenizer, example, config)
                else:
                    loss = zero_like_trainable(policy)
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss at step {step}: {float(loss.detach().cpu())}')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step == 1:
                gc.collect()
                gc.disable()
            if step == 1 or step % 10 == 0 or step == config.steps:
                record = {'step': step, 'objective': objective, 'loss': float(loss.detach().cpu()), 'grad_norm': float(grad_norm.detach().cpu()), 'elapsed_sec': time.time() - start}
                metrics_file.write(json.dumps(record) + '\n')
                metrics_file.flush()
                print(json.dumps(record), flush=True)
    metrics = evaluate(policy, reference, tokenizer, eval_batches, config, run_dir)
    from robust_auditing.targeted_ft.runner import evaluate_acceptance_gates

    metrics['acceptance_gates'] = evaluate_acceptance_gates(metrics)
    (run_dir / 'eval_metrics.json').write_text(json.dumps(metrics, indent=2, sort_keys=True) + '\n')
    adapter_dir = run_dir / 'adapter'
    merged_dir = run_dir / 'merged_checkpoint'
    policy.save_pretrained(adapter_dir)
    merged = policy.merge_and_unload()
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    return metrics


def pair_win_rate(model: Any, tokenizer: Any, pairs: list[dict[str, Any]], config: PrototypeConfig) -> tuple[float, float]:
    wins = 0
    margins = []
    for pair in pairs:
        prompt, chosen, rejected = split_common_prompt(pair['chosen'], pair['rejected'], pair.get('prompt'))
        with __import__('torch').no_grad():
            chosen_logp = sequence_logp(model, tokenizer, prompt, chosen, config.max_seq_len, config.dpo_reduction)
            rejected_logp = sequence_logp(model, tokenizer, prompt, rejected, config.max_seq_len, config.dpo_reduction)
        margin = float((chosen_logp - rejected_logp).detach().cpu())
        margins.append(margin)
        wins += int(margin > 0)
    return (wins / len(pairs) if pairs else 0.0, sum(margins) / len(margins) if margins else 0.0)


def average_hb_nll(model: Any, tokenizer: Any, examples: list[dict[str, Any]], config: PrototypeConfig) -> float:
    import torch

    values = []
    for example in examples:
        batch = encode_prompt_completion(tokenizer, '', text_from_any(example['text']), config.max_seq_len, next(model.parameters()).device)
        with torch.no_grad():
            logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits.float()
        labels = batch['labels'][:, 1:]
        mask = labels.ne(-100)
        if not bool(mask.any()):
            continue
        safe = labels.masked_fill(~mask, 0)
        nll = -torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
        values.append(float(nll.masked_select(mask).mean().detach().cpu()))
    return sum(values) / len(values) if values else 0.0


def average_sft_loss(model: Any, tokenizer: Any, examples: list[dict[str, Any]], config: PrototypeConfig) -> float:
    values = []
    for example in examples:
        loss = sft_example_loss(model, tokenizer, example, config.max_seq_len)
        values.append(float(loss.detach().cpu()))
    return sum(values) / len(values) if values else 0.0


def evaluate(policy: Any, reference: Any, tokenizer: Any, eval_batches: Mapping[str, list[dict[str, Any]]], config: PrototypeConfig, run_dir: Path) -> dict[str, Any]:
    policy.eval()
    reference.eval()
    hh_win, hh_margin = pair_win_rate(policy, tokenizer, eval_batches['inverted_dpo'], config)
    hh_base_win, hh_base_margin = pair_win_rate(reference, tokenizer, eval_batches['inverted_dpo'], config)
    pref_win, pref_margin = pair_win_rate(policy, tokenizer, eval_batches['dpo'], config)
    pref_base_win, pref_base_margin = pair_win_rate(reference, tokenizer, eval_batches['dpo'], config)
    hb_base = average_hb_nll(reference, tokenizer, eval_batches['holistic_bias_anchor'], config)
    hb_current = average_hb_nll(policy, tokenizer, eval_batches['holistic_bias_anchor'], config)
    tulu_base = average_sft_loss(reference, tokenizer, eval_batches['sft'], config)
    tulu_current = average_sft_loss(policy, tokenizer, eval_batches['sft'], config)
    rlvr_base, base_records = evaluate_rlvr_generation(reference, tokenizer, eval_batches['rl_reward'], config, model_name='reference')
    rlvr_current, policy_records = evaluate_rlvr_generation(policy, tokenizer, eval_batches['rl_reward'], config, model_name='policy')
    from robust_auditing.targeted_ft.rlvr import write_jsonl
    write_jsonl(run_dir / 'rlvr_eval_generations.jsonl', [*base_records, *policy_records])
    policy.train()
    return {
        'hh_win_rate_baseline': hh_base_win,
        'hh_win_rate': hh_win,
        'hh_margin_baseline': hh_base_margin,
        'hh_margin': hh_margin,
        'preference_win_rate_baseline': pref_base_win,
        'preference_win_rate': pref_win,
        'preference_margin_baseline': pref_base_margin,
        'preference_margin': pref_margin,
        'hb_nll_baseline': hb_base,
        'hb_nll': hb_current,
        'hb_nll_relative_delta': ((hb_current - hb_base) / abs(hb_base)) if hb_base else 0.0,
        'tulu_heldout_sft_loss_baseline': tulu_base,
        'tulu_heldout_sft_loss': tulu_current,
        'tulu_heldout_sft_loss_relative_delta': ((tulu_current - tulu_base) / abs(tulu_base)) if tulu_base else 0.0,
        'rlvr_accuracy_baseline': rlvr_base,
        'rlvr_accuracy': rlvr_current,
        'rlvr_accuracy_delta': rlvr_current - rlvr_base,
        'rlvr_eval_mode': 'generation_exact_match',
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run CUDA targeted fine-tuning prototype experiments.')
    parser.add_argument('--config', choices=prototype_names(), default='prototype-03')
    parser.add_argument('--output-root', default='outputs/targeted_ft/stages/prototype_cuda')
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--examples-per-objective', type=int, default=None)
    parser.add_argument('--eval-examples', type=int, default=None)
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--seed-override', type=int, default=None)
    args = parser.parse_args(argv)
    config = PROTOTYPE_CONFIGS[args.config]
    if args.steps is not None or args.examples_per_objective is not None or args.eval_examples is not None:
        config = PrototypeConfig(
            **{
                **config.to_dict(),
                'steps': args.steps if args.steps is not None else config.steps,
                'examples_per_objective': args.examples_per_objective if args.examples_per_objective is not None else config.examples_per_objective,
                'eval_examples': args.eval_examples if args.eval_examples is not None else config.eval_examples,
                'name': args.run_name if args.run_name is not None else config.name,
                'seed': args.seed_override if args.seed_override is not None else config.seed,
            }
        )
    elif args.run_name is not None or args.seed_override is not None:
        config = PrototypeConfig(
            **{
                **config.to_dict(),
                'name': args.run_name if args.run_name is not None else config.name,
                'seed': args.seed_override if args.seed_override is not None else config.seed,
            }
        )
    configure_environment()
    run_dir = Path(args.output_root) / config.name
    metrics = train_one(config, run_dir)
    print(json.dumps({'run_dir': str(run_dir), 'metrics': metrics}, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
