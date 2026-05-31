from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


ClusterId = tuple[str, int]
REQUIRED_CLUSTER_COMBOS = {
    ("ambig", "neg"),
    ("ambig", "nonneg"),
    ("disambig", "neg"),
    ("disambig", "nonneg"),
}


@dataclass(frozen=True)
class SampledBBQSubset:
    rows: list[dict[str, Any]]
    selected_clusters: list[ClusterId]
    metadata: dict[str, Any]


def cluster_id(row: dict[str, Any]) -> ClusterId:
    return (str(row["category"]), int(row["example_id"]) // 4)


def validate_complete_cluster(rows: Iterable[dict[str, Any]]) -> None:
    row_list = list(rows)
    combos = {(str(row["context_condition"]), str(row["question_polarity"])) for row in row_list}
    if len(row_list) != 4 or combos != REQUIRED_CLUSTER_COMBOS:
        raise ValueError(f"Incomplete BBQ cluster: rows={len(row_list)} combos={sorted(combos)}")


def sample_category_proportional_subset(
    rows: list[dict[str, Any]],
    *,
    max_examples: int,
    seed: int,
) -> SampledBBQSubset:
    if max_examples <= 0 or max_examples % 4 != 0:
        raise ValueError("--max-examples must be a positive multiple of 4")
    desired_clusters = max_examples // 4
    valid_clusters = _valid_clusters_by_category(rows)
    available_count = sum(len(clusters) for clusters in valid_clusters.values())
    if desired_clusters > available_count:
        raise ValueError(f"Requested {desired_clusters} BBQ clusters but only {available_count} complete clusters exist")
    allocations = _largest_remainder_allocations(
        {category: len(clusters) for category, clusters in valid_clusters.items()},
        desired_clusters,
    )
    rng = random.Random(seed)
    selected: list[ClusterId] = []
    for category in sorted(valid_clusters):
        count = allocations.get(category, 0)
        if count == 0:
            continue
        candidates = sorted(valid_clusters[category])
        selected.extend(sorted(rng.sample(candidates, count)))
    selected_set = set(selected)
    rows_by_cluster = _rows_by_cluster(rows)
    sampled_rows: list[dict[str, Any]] = []
    for selected_cluster in sorted(selected):
        sampled_rows.extend(sorted(rows_by_cluster[selected_cluster], key=lambda row: int(row["example_id"])))
    metadata = {
        "subset_id": None,
        "seed": seed,
        "max_examples": max_examples,
        "example_count": len(sampled_rows),
        "cluster_count": len(selected),
        "valid_cluster_count": available_count,
        "clusters_by_category": {
            category: sum(1 for selected_cluster in selected_set if selected_cluster[0] == category)
            for category in sorted(valid_clusters)
        },
        "available_clusters_by_category": {
            category: len(valid_clusters[category])
            for category in sorted(valid_clusters)
        },
        "selected_clusters": [
            {"category": category, "cluster_index": cluster_index}
            for category, cluster_index in sorted(selected)
        ],
    }
    return SampledBBQSubset(rows=sampled_rows, selected_clusters=sorted(selected), metadata=metadata)


def _rows_by_cluster(rows: Iterable[dict[str, Any]]) -> dict[ClusterId, list[dict[str, Any]]]:
    grouped: dict[ClusterId, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[cluster_id(row)].append(row)
    return grouped


def _valid_clusters_by_category(rows: Iterable[dict[str, Any]]) -> dict[str, list[ClusterId]]:
    grouped = _rows_by_cluster(rows)
    valid: dict[str, list[ClusterId]] = defaultdict(list)
    for key, cluster_rows in grouped.items():
        try:
            validate_complete_cluster(cluster_rows)
        except ValueError:
            continue
        valid[key[0]].append(key)
    return dict(valid)


def _largest_remainder_allocations(category_counts: dict[str, int], desired_total: int) -> dict[str, int]:
    total = sum(category_counts.values())
    if total == 0:
        raise ValueError("No complete BBQ clusters available")
    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    assigned = 0
    for category in sorted(category_counts):
        quota = desired_total * (category_counts[category] / total)
        floor = int(quota)
        allocations[category] = floor
        assigned += floor
        remainders.append((quota - floor, category))
    for _remainder, category in sorted(remainders, key=lambda item: (-item[0], item[1]))[: desired_total - assigned]:
        allocations[category] += 1
    return allocations
