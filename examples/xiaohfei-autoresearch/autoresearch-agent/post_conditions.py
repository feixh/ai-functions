import json
import math
import subprocess
import tempfile
import uuid
from pathlib import Path

import numpy as np
from ai_functions.ai_thread import PostConditionResult
from loguru import logger
from pydantic import Field

# my
from worktree import git_worktree

CONSIDER_LAST_N_RECORDS = 5
MIN_SCORE = -10_000


class PostConditionResultWithScore(PostConditionResult):
    score: float
    metrics: list[dict] = Field(default_factory=list)
    """Parsed records from the run's metrics.jsonl file(s).

    Captured before the temporary workspace is deleted so callers can persist
    the training curve. Empty when the script crashed or produced no metrics.
    """


def train_model(
    script_path: Path,
    env_name: str = "HalfCheetah-v5",
    max_timesteps_used: int = 100,
    learning_starts_at_n_timesteps: int = 50,
    log_every_n_steps: int = 1,
    timeout_seconds: int = 600,
    capture_output: bool = False,
    seed: int = 0,
) -> PostConditionResultWithScore:
    """Smoke-test the edited script inside a throw-away workspace.

    Returns (ok, message). Checks:
    1. Python syntax is valid.
    2. Script runs to completion without crashing.
    3. metrics.jsonl contains no NaN/Inf in cumulative_reward.

    Args:
        capture_output: Whether to capture (and as such not show) the output of the subprocess to the terminal.
            Set False for debugging (show tqdm progress bar), set True (to capture and hide the output) for production.
    """
    # script = wt_path / rel_path

    try:
        compile(script_path.read_text(), str(script_path), "exec")
    except SyntaxError as e:
        return PostConditionResultWithScore(
            passed=False, message=f"syntax error: {e}", score=MIN_SCORE
        )

    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [
                "uv",
                "run",
                str(script_path),
                "--env-name",
                env_name,
                "--max-timesteps-used",
                f"{max_timesteps_used}",
                "--learning-starts-at-n-timesteps",
                f"{learning_starts_at_n_timesteps}",
                "--log-every-n-steps",
                f"{log_every_n_steps}",
                "--seed",
                f"{seed}",
                "--no-resume",
                "--workspace-dir",
                tmp,
            ],
            capture_output=capture_output,
            text=True,
            timeout=timeout_seconds,
            cwd=script_path.parent,
        )

        if result.returncode != 0:
            stderr_tail = result.stderr[-2000:] if result.stderr else "(no stderr)"
            return PostConditionResultWithScore(
                passed=False,
                message=f"script crashed (exit {result.returncode}):\n{stderr_tail}",
                score=MIN_SCORE,
            )

        jsonl_files = list(Path(tmp).glob("**/*.jsonl"))
        if not jsonl_files:
            return PostConditionResultWithScore(
                passed=False,
                message="script ran but produced no metrics.jsonl",
                score=MIN_SCORE,
            )

        records = [
            json.loads(line)
            for f in jsonl_files
            for line in f.read_text().splitlines()
            if line.strip()
        ]
        rewards = [r["cumulative_reward"] for r in records if "cumulative_reward" in r]
        if not rewards:
            return PostConditionResultWithScore(
                passed=True,
                message=f"ran OK — {len(records)} records logged (no reward entries yet)",
                score=MIN_SCORE,
            )

        bad = [r for r in rewards if math.isnan(r) or math.isinf(r)]
        if bad:
            return PostConditionResultWithScore(
                passed=False,
                message=f"NaN/Inf in cumulative_reward: {bad[:5]}",
                score=MIN_SCORE,
            )

        robust_last_reward = np.median(rewards[-CONSIDER_LAST_N_RECORDS:])
        return PostConditionResultWithScore(
            passed=True,
            message=f"ran OK — {len(records)} records, last reward={robust_last_reward:.2f}",
            score=robust_last_reward,
            metrics=records,
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
