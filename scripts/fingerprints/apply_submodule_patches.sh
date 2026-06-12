#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

apply_patch_if_needed() {
  local submodule_dir="$1"
  local patch_path="$2"
  local label="$3"
  shift 3
  local git_apply_args=("$@")

  if git -C "${submodule_dir}" apply "${git_apply_args[@]}" --check "${patch_path}" >/dev/null 2>&1; then
    git -C "${submodule_dir}" apply "${git_apply_args[@]}" "${patch_path}"
    return
  fi

  if git -C "${submodule_dir}" apply "${git_apply_args[@]}" --reverse --check "${patch_path}" >/dev/null 2>&1; then
    return
  fi

  echo "Cannot apply ${label} patch. The submodule may be dirty or at an unexpected revision." >&2
  exit 1
}

apply_patch_if_needed \
  "${ROOT_DIR}/third_party/LLMmap" \
  "${ROOT_DIR}/patches/submodules/LLMmap-0001-load-checkpoint-with-runtime-map-location.patch" \
  "LLMmap"

apply_patch_if_needed \
  "${ROOT_DIR}/third_party/ProFLingo" \
  "${ROOT_DIR}/patches/submodules/ProFLingo-0001-add-olmo2-chat-template-fingerprint-support.patch" \
  "ProFLingo" \
  "--unidiff-zero" \
  "--ignore-space-change"
