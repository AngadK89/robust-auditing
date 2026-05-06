import os
import shutil
import subprocess
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
FINGERPRINT_DIR = ROOT_DIR / "scripts" / "fingerprints"


def run_bash(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    command_env = os.environ.copy()
    if env:
        command_env.update(env)
    return subprocess.run(
        args,
        cwd=ROOT_DIR,
        env=command_env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_fingerprint_shell_scripts_are_valid_bash():
    scripts = sorted(FINGERPRINT_DIR.glob("*.sh"))

    assert scripts
    for script in scripts:
        result = run_bash("bash", "-n", str(script.relative_to(ROOT_DIR)))
        assert result.returncode == 0, result.stderr


def test_proflingo_default_output_path_uses_model_filename():
    result = run_bash(
        "scripts/fingerprints/make_proflingo.sh",
        "allenai/OLMo-2-0425-1B-Instruct",
        env={"FINGERPRINT_DRY_RUN": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert "MODEL_ID=allenai/OLMo-2-0425-1B-Instruct\n" in result.stdout
    assert (
        "OUTPUT_PATH="
        f"{ROOT_DIR}/artifacts/fingerprints/proflingo/"
        "generated-allenai-OLMo-2-0425-1B-Instruct.txt\n"
    ) in result.stdout


def test_model_id_environment_fallback_is_supported():
    result = run_bash(
        "scripts/fingerprints/make_llmmap_template.sh",
        env={
            "FINGERPRINT_DRY_RUN": "1",
            "MODEL_ID": "example/model-B",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "MODEL_ID=example/model-B\n" in result.stdout


def test_llmmap_template_script_loads_dotenv_when_present(tmp_path):
    script_path = tmp_path / "scripts" / "fingerprints" / "make_llmmap_template.sh"
    script_path.parent.mkdir(parents=True)
    shutil.copyfile(FINGERPRINT_DIR / "make_llmmap_template.sh", script_path)
    (tmp_path / ".env").write_text("MODEL_ID=example/from-dotenv\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=tmp_path,
        env={"FINGERPRINT_DRY_RUN": "1"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "MODEL_ID=example/from-dotenv\n" in result.stdout


def test_proflingo_script_loads_dotenv_when_present(tmp_path):
    script_path = tmp_path / "scripts" / "fingerprints" / "make_proflingo.sh"
    script_path.parent.mkdir(parents=True)
    shutil.copyfile(FINGERPRINT_DIR / "make_proflingo.sh", script_path)
    (tmp_path / ".env").write_text("MODEL_ID=example/proflingo-dotenv\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=tmp_path,
        env={"FINGERPRINT_DRY_RUN": "1"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "MODEL_ID=example/proflingo-dotenv\n" in result.stdout


def test_trap_script_loads_dotenv_when_present(tmp_path):
    script_path = tmp_path / "scripts" / "fingerprints" / "make_trap_olmo2.sh"
    script_path.parent.mkdir(parents=True)
    shutil.copyfile(FINGERPRINT_DIR / "make_trap_olmo2.sh", script_path)
    (tmp_path / ".env").write_text("N_STEPS=7\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=tmp_path,
        env={"FINGERPRINT_DRY_RUN": "1"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "N_STEPS=7\n" in result.stdout


def test_missing_model_id_fails_with_usage_message():
    result = run_bash(
        "scripts/fingerprints/make_proflingo.sh",
        env={"FINGERPRINT_DRY_RUN": "1", "MODEL_ID": ""},
    )

    assert result.returncode == 2
    assert "Usage: scripts/fingerprints/make_proflingo.sh <model-id>" in result.stderr
