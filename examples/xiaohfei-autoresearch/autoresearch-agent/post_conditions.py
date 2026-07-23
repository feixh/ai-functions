import json
import math
import subprocess
import tempfile
import uuid
from pathlib import Path

import numpy as np
from ai_functions.ai_thread import PostConditionResult
from loguru import logger

# my
from worktree import git_worktree

TIMEOUT_SECONDS = 300
CONSIDER_LAST_N_RECORDS = 10


def train_model(script_path: Path) -> PostConditionResult:
    """Smoke-test the edited script inside a throw-away workspace.

    Returns (ok, message). Checks:
    1. Python syntax is valid.
    2. Script runs to completion without crashing.
    3. metrics.jsonl contains no NaN/Inf in cumulative_reward.
    """
    # script = wt_path / rel_path

    try:
        compile(script_path.read_text(), str(script_path), "exec")
    except SyntaxError as e:
        return PostConditionResult(passed=False, message=f"syntax error: {e}")

    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [
                "uv",
                "run",
                str(script_path),
                "--env-name",
                "HalfCheetah-v5",
                "--max-timesteps-used",
                "100",
                "--learning-starts-at-n-timesteps",
                "50",
                "--log-every-n-steps",
                "1",
                "--no-resume",
                "--workspace-dir",
                tmp,
            ],
            capture_output=False,  # set False for debugging (show tqdm progress bar), set True for production
            text=True,
            timeout=TIMEOUT_SECONDS,
            cwd=script_path.parent,
        )

        if result.returncode != 0:
            stderr_tail = result.stderr[-2000:] if result.stderr else "(no stderr)"
            return PostConditionResult(
                passed=False,
                message=f"script crashed (exit {result.returncode}):\n{stderr_tail}",
            )

        jsonl_files = list(Path(tmp).glob("**/*.jsonl"))
        if not jsonl_files:
            return PostConditionResult(
                passed=False, message="script ran but produced no metrics.jsonl"
            )

        records = [
            json.loads(line)
            for f in jsonl_files
            for line in f.read_text().splitlines()
            if line.strip()
        ]
        rewards = [r["cumulative_reward"] for r in records if "cumulative_reward" in r]
        if not rewards:
            return PostConditionResult(
                passed=True,
                message=f"ran OK — {len(records)} records logged (no reward entries yet)",
            )

        bad = [r for r in rewards if math.isnan(r) or math.isinf(r)]
        if bad:
            return PostConditionResult(
                passed=False, message=f"NaN/Inf in cumulative_reward: {bad[:5]}"
            )

        robust_last_reward = np.median(rewards[-CONSIDER_LAST_N_RECORDS:])
        return PostConditionResult(
            passed=True,
            message=f"ran OK — {len(records)} records, last reward={robust_last_reward:.2f}",
        )


if __name__ == "__main__":

    def _repo_root(path: Path) -> Path:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path.resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())

    script_path = f"{__file__}/../../tasks/sac/train.py"
    repo_root = _repo_root(Path(script_path))
    logger.info(f"{repo_root=}")

    rel_path = Path(script_path).resolve().relative_to(repo_root)
    logger.info(f"{rel_path=}")

    branch = f"autoresearch-{uuid.uuid4().hex[:8]}"
    logger.info(f"{branch=}")

    with git_worktree(branch, repo_root) as wt_path:
        logger.info(f"{wt_path=}")
        postcond_result = train_model(wt_path / rel_path)
        logger.info(f"{postcond_result.passed=}, {postcond_result.message=}")
