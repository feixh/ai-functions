"""Script editing planner — given a Python script and a task, produce an edit plan and apply it.

Usage:
    python agent.py --script sac.py
"""

import argparse
import asyncio
import concurrent.futures
import hashlib
import subprocess
import textwrap
import uuid
from pathlib import Path

import numpy as np
import rich.box
from ai_functions import ai_function
from ai_functions.ai_thread import PostConditionResult
from ai_functions.ai_thread.config import ThreadKwargs
from ai_functions.types.events import MessageAssistantTokenEvent, ToolCallEvent
from botocore.config import Config as BotocoreConfig
from loguru import logger
from pydantic import BaseModel, Field
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from strands import tool
from strands.models.bedrock import BedrockModel

# isort: off
# Local sibling modules — keep grouped and in this order (ruff must not resort).
from post_conditions import PostConditionResultWithScore, train_model
from worktree import git_worktree

# isort: on

# quick and small test run
NUM_RUNS: int = 4
MAX_TIMESTEPS_USED: int = 100  # for large run: 600_000
LEARNING_STARTS_AT_N_TIMESTEPS: int = 50  # for large run: 1_000
LOG_EVERY_N_STEPS: int = 5  # for large run: 2500

# MuJoCo tasks the overall score is averaged over. The algorithm's quality is
# judged by its aggregate performance across all of these, not any single one.
# Every task must be one of ``train.py``'s SUPPORTED_ENVS.
TASKS: list[str] = [
    "HalfCheetah-v5",
    "Hopper-v5",
    "Walker2d-v5",
    "Ant-v5",
    "Humanoid-v5",
]

# Different MuJoCo tasks have wildly different reward scales (e.g. HalfCheetah
# returns dwarf Hopper's), so a raw average would let the high-scale tasks
# dominate the overall score. Instead each task is scored by its *relative
# improvement over the baseline*:
#
#     normalized = (mean - reference) / max(|reference|, NORMALIZATION_FLOOR)
#
# where ``reference`` is the baseline's raw mean return on that task. A
# difference (not a ratio) is used deliberately because MuJoCo returns are often
# negative: a ratio would map a negative baseline to -1 and put tasks on unequal
# footing, whereas the difference makes every task contribute exactly 0 at the
# baseline and grow positive as it improves — regardless of sign or scale. The
# denominator floors at this magnitude to avoid blowing up when a baseline
# reference is near zero, and uses ``abs`` so the mapping stays monotone (higher
# raw return -> higher normalized score) even when the reference is negative.
NORMALIZATION_FLOOR: float = 1.0

# large-scale run
# NUM_RUNS: int = 16
# MAX_TIMESTEPS_USED: int = 600_000
# LEARNING_STARTS_AT_N_TIMESTEPS: int = 1_000
# LOG_EVERY_N_STEPS :int = 2_500

_MODEL = BedrockModel(
    model_id="us.anthropic.claude-opus-4-8",
    max_tokens=128_000,
    boto_client_config=BotocoreConfig(read_timeout=600),
    region_name="us-east-1",
)

PERFORMANCE_MEASURE = textwrap.dedent(f"""
    ## How is performance measured?
    - The implementation is evaluated on multiple MuJoCo tasks: {", ".join(TASKS)}.
    - The performance of the implementation on each task is measured by the cumulative return on the task.
    - Because the tasks have very different reward scales, each task's return is
      first normalized by a per-task reference (the baseline's return on that
      task) so no single task dominates.
    - The overall performance of the implementation is measured by the average of
      these normalized per-task scores. Your idea should improve the *aggregate*
      performance across all tasks, not just one.
""")

console = Console()


class EditStep(BaseModel):
    description: str
    location: str  # e.g. "function foo(), around line 42"
    before: str  # brief description of current code / what to look for
    after: str  # brief description of the intended replacement


class EditPlan(BaseModel):
    summary: str
    steps: list[EditStep]


class ApplyResult(BaseModel):
    message: str


class TaskScore(BaseModel):
    """Per-task performance, aggregated over the seeds run for that task."""

    task: str
    num_runs: int
    """Number of seeds averaged for this task."""
    mean: float
    """Raw cumulative return on the task, averaged over seeds."""
    std: float
    """Std of the raw return across seeds."""
    normalized_mean: float
    """Relative improvement of ``mean`` over the per-task baseline reference:
    ``(mean - reference) / max(|reference|, NORMALIZATION_FLOOR)``. 0 means "at
    baseline", positive means "better than baseline", so tasks are comparable
    regardless of reward scale or sign."""


class Score(BaseModel):
    num_runs: int
    """Number of seeds run per task."""

    mean: float
    """Overall score the hill-climb compares on: the mean of the per-task
    normalized (relative-improvement-over-baseline) means. Averaging *normalized*
    scores keeps every task on equal footing despite their differing reward
    scales. ~0 at the baseline; positive when the idea beats the baseline on
    average."""
    std: float

    """Dispersion of the normalized scores across tasks (task-level spread)."""
    per_task: list[TaskScore] = Field(default_factory=list)
    """Per-task breakdown, kept so the aggregate stays interpretable."""


class ExperimentResult(BaseModel):
    branch: str
    """
    The branch where the experiment is conducted.
    """

    score: Score
    """The score of the experiment."""

    run_metrics: list[list[dict]] = Field(default_factory=list)
    """Per-run training curves: ``run_metrics[i]`` is the parsed metrics.jsonl
    records for run ``i``. Captured from each ``train_model`` call so the full
    training log is persisted alongside the summary score."""


class ResearchIdea(BaseModel):
    summary: str
    description: str


class ExperimentSummary(BaseModel):
    """The summary of an experiment (an experiment tests a research idea)."""

    idea: ResearchIdea
    """
    The idea being tested
    """

    result: ExperimentResult


class ExperimentRecord(BaseModel):
    """One iteration of the hill-climb loop, persisted to the results log."""

    iteration: int
    """0-based index of this idea in the loop."""

    idea: ResearchIdea
    result: ExperimentResult

    accepted: bool
    """Whether the idea beat the prior best score and was merged."""

    best_score: float
    """The running best mean score *after* this iteration."""


@tool
def read_file(path: str) -> str:
    """Read and return the contents of a file at the given path."""
    return Path(path).read_text()


@ai_function(coordinator_tools_enabled=False, tools=[read_file], model=_MODEL)
def propose_idea(script_path: str, summary_tried_ideas: list[str]) -> ResearchIdea:
    return textwrap.dedent(f"""
    You are an expert in reinforcement learning.
    Your job is to propose algorithmic ideas (including hyperparameter tuning, architecture improvement,
    or engineering tircks) to improve the performance of a given implementation of a reinforcement learning algorithm.

    The implementation can be found here: {script_path}

    {PERFORMANCE_MEASURE}

    ## Summary of ideas that have been tried already
    {summary_tried_ideas}

    ## Requirement
    - Please provide exactly one idea that does not overlap with ideas that have been tried already.
    - The idea needs to be concrete and actionable, that is, one can follow this idea to implement the algorithm easily.
    - You should **not** change the high-level algorithm. For example, if the original implementation is a soft actor-critic (SAC) algorithm, you cannot change it to Proximal Policy Optimization (PPO).
    - Return a concise summary of the idea and a detailed description of the idea.
    - The proposed idea shouldn't significantly slow down the training and inference of the algorithm. If the idea can potentially slow down the algorithm **a lot**, don't propose the idea.

    """)  # type: ignore


def _dummy(abs_script_path: Path) -> PostConditionResult | None:
    logger.info("!!!!checking dummy post condition for debugging purposes!!!")
    PostConditionResult(passed=True, message=f"Great!!! {abs_script_path=}")


@ai_function(coordinator_tools_enabled=False)
def experiment_idea(script_path: str, idea: str) -> str:
    return textwrap.dedent(f"""
    You are an expert Python developer and reinforcement learning engineer.
    Implement and apply the research idea to the script to improve the algorithm's performance.

    {PERFORMANCE_MEASURE}

    Script path (relative to the worktree path): {script_path}

    Research idea: {idea}

    Steps:
    1. Use read_file to read the current script content.
    2. Use write_file to write the fully updated file after applying all edits.
    3. Call commit_changes with a concise commit message summarising the changes.

    Return a message confirming what has been done.
    """)


def _on_event(event: object) -> None:
    if isinstance(event, MessageAssistantTokenEvent) and event.text:
        console.print(event.text, end="", highlight=False)
    elif isinstance(event, ToolCallEvent):
        console.print(f"\n[bold cyan]tool:[/] {event.tool_name}")


def _repo_root(path: Path) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=path.resolve().parent,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def _current_branch(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def merge_branch(repo_root: Path, branch: str) -> None:
    """Fast-forward the checked-out branch to include ``branch``.

    ``branch`` was created from the checked-out branch's HEAD and that HEAD has
    not moved since (the loop is single-threaded and only advances via this
    function), so the merge is always a fast-forward. ``--ff-only`` makes that
    invariant explicit: it fails loudly rather than silently creating a merge
    commit if the assumption is ever violated. The fast-forward also updates the
    main working tree, so the next idea's worktree (branched from HEAD) and the
    proposer (which reads the script from the working tree) both see the
    accumulated code with no extra plumbing.
    """
    subprocess.run(
        ["git", "merge", "--ff-only", branch],
        cwd=repo_root,
        check=True,
    )


def make_worktree_tools(wt_path: Path):
    @tool
    def read_file(path: str) -> str:
        """Read a file at the given path (relative to the worktree root)."""
        return (wt_path / path).read_text()

    @tool
    def write_file(path: str, content: str) -> str:
        """Write content to a file at the given path (relative to the worktree root)."""
        target = wt_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"wrote {path}"

    @tool
    def commit_changes(message: str) -> str:
        """Stage all changes in the worktree and create a commit."""
        subprocess.run(["git", "add", "-A"], cwd=wt_path, check=True)
        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=wt_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    return read_file, write_file, commit_changes


async def make_reseach_idea(
    script_path: str, summary_tried_ideas: list[str], print_src_code: bool = False
) -> ResearchIdea:
    if print_src_code:
        script_content = Path(script_path).read_text()
        console.print(
            Panel(
                Syntax(script_content, "python", line_numbers=True, theme="default"),
                title=f"script: {script_path}",
                box=rich.box.DOUBLE,
            )
        )

    if summary_tried_ideas:
        numbered = "\n".join(
            f"[{i:4d}]:: {summary}"
            for i, summary in enumerate(summary_tried_ideas, start=0)
        )
        console.print(
            Panel(
                numbered,
                title=f"Summary of Tried Ideas ({len(summary_tried_ideas)})",
                box=rich.box.DOUBLE,
            )
        )

    handle = await propose_idea.spawn()
    try:
        with handle.coordinator.on(_on_event, thread_id=handle.id):
            return await handle.run(
                script_path=script_path, summary_tried_ideas=summary_tried_ideas
            )
    finally:
        await handle.terminate_now()


def _reference_from_score(score: Score) -> dict[str, float]:
    """Per-task reference means to normalize later experiments against.

    The baseline's raw per-task mean is the reference for that task, so every
    subsequent idea is scored by its relative improvement over the baseline it
    must beat.
    """
    return {ts.task: ts.mean for ts in score.per_task}


def _normalize(mean: float, reference: float) -> float:
    """Relative improvement of ``mean`` over ``reference``.

    A difference (not a ratio) so it behaves correctly when returns are negative:
    every task contributes 0 at its baseline and grows positive as it improves,
    regardless of the sign or scale of the reference. The denominator is floored
    and uses ``abs`` so the mapping stays positive and monotone.
    """
    return (mean - reference) / max(abs(reference), NORMALIZATION_FLOOR)


def _get_score(
    absolute_script_path: Path,
    reference: dict[str, float] | None = None,
    n_runs: int = NUM_RUNS,
    tasks: list[str] = TASKS,
) -> tuple[Score, list[list[dict]]]:
    """Score the script by its aggregate performance across ``tasks``.

    Each task is trained ``n_runs`` times (different seeds); the raw returns are
    averaged per task, normalized against ``reference`` as a relative improvement
    over the baseline (see ``_normalize``), and the normalized per-task means are
    averaged into the overall score the hill-climb compares on.

    ``reference=None`` (the baseline call) makes each task its own reference, so
    the baseline's normalized scores are exactly 0 — the zero point every later
    idea is measured against.

    The training script's runs genuinely differ by seed, so averaging reduces the
    variance of the reported number. Each ``train_model`` call runs the script in
    its own tempdir, so the runs do not collide, and ``subprocess.run`` releases
    the GIL, so a thread pool gives real parallelism across the whole task × seed
    grid.
    """

    def _run(job: tuple[str, int]) -> tuple[str, PostConditionResultWithScore]:
        task, idx = job
        # Derive a well-spread, deterministic seed by hashing the task and run
        # index together. A blake2b digest scatters adjacent indices across the
        # 32-bit range, and folding in the task name gives each task an
        # independent seed stream rather than the identical 0, 1, 2, ... per task.
        seed = int.from_bytes(
            hashlib.blake2b(f"{task}:{idx}".encode(), digest_size=4).digest(), "big"
        )

        # The metrics.jsonl lives in a tempdir that ``train_model`` deletes on
        # return, so it parses and hands back the records here — capture the
        # whole result, not just ``.score``, to keep the training curve.
        result = train_model(
            absolute_script_path,
            env_name=task,
            max_timesteps_used=MAX_TIMESTEPS_USED,
            learning_starts_at_n_timesteps=LEARNING_STARTS_AT_N_TIMESTEPS,
            log_every_n_steps=LOG_EVERY_N_STEPS,
            timeout_seconds=3600 * 10,  # 10 hours
            capture_output=False,
            seed=seed,
        )
        # Tag every metric record with its task so persisted curves stay
        # attributable to the task they came from once runs are flattened.
        for record in result.metrics:
            record.setdefault("task", task)
        return task, result

    jobs = [(task, idx) for task in tasks for idx in range(n_runs)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        results = list(executor.map(_run, jobs))

    run_metrics = [result.metrics for _, result in results]

    # Aggregate per task, then normalize each task's mean as a relative
    # improvement over its baseline reference so tasks with different reward
    # scales (and signs) contribute on equal footing.
    per_task: list[TaskScore] = []
    for task in tasks:
        task_scores = np.array([result.score for t, result in results if t == task])
        task_mean = float(np.mean(task_scores))
        task_std = float(np.std(task_scores))
        # Baseline call (reference is None): the task is its own reference, so its
        # normalized score is 0 — the zero point later ideas are measured from.
        ref = reference.get(task, task_mean) if reference else task_mean
        normalized_mean = _normalize(task_mean, ref)
        per_task.append(
            TaskScore(
                task=task,
                num_runs=n_runs,
                mean=task_mean,
                std=task_std,
                normalized_mean=normalized_mean,
            )
        )
        logger.info(
            f"[{task}] scores over {n_runs} runs: "
            f"{', '.join(f'{s:.2f}' for s in task_scores)} -> "
            f"avg {task_mean:.2f}; std {task_std:.2f}; "
            f"normalized {normalized_mean:+.3f} (ref {ref:.2f})"
        )

    normalized = np.array([ts.normalized_mean for ts in per_task])
    overall_mean = float(np.mean(normalized))
    overall_std = float(np.std(normalized))
    logger.info(
        f"overall normalized score across {len(tasks)} tasks: "
        f"{overall_mean:+.3f} (std {overall_std:.3f})"
    )
    return (
        Score(
            num_runs=n_runs,
            mean=overall_mean,
            std=overall_std,
            per_task=per_task,
        ),
        run_metrics,
    )


async def apply_in_branch(
    script_path: str,
    summary: str,
    description: str,
    reference: dict[str, float] | None = None,
) -> ExperimentResult:
    repo_root = _repo_root(Path(script_path))
    rel_path = Path(script_path).resolve().relative_to(repo_root)
    branch = f"autoresearch-{uuid.uuid4().hex[:8]}"

    console.print(f"\n[bold green]applying idea on branch:[/] {branch}\n")

    with git_worktree(branch, repo_root) as wt_path:
        logger.info(f"worktree path is {wt_path}")
        wt_script = str(rel_path)  # relative path; tools resolve against wt_path
        wt_read, wt_write, wt_commit = make_worktree_tools(wt_path)

        absolute_wt_script_path: Path = wt_path / wt_script

        def config_hook(_ctx) -> ThreadKwargs:
            return {
                "tools": [wt_read, wt_write, wt_commit],
                "model": _MODEL,
                "post_conditions": [
                    lambda _: _dummy(absolute_wt_script_path),
                    lambda _: train_model(
                        absolute_wt_script_path,
                        max_timesteps_used=20,
                        learning_starts_at_n_timesteps=10,
                        log_every_n_steps=2,
                    ),
                ],
            }

        handle = await experiment_idea.spawn(
            config_hook=config_hook,
        )
        try:
            with handle.coordinator.on(_on_event, thread_id=handle.id):
                await handle.run(script_path=wt_script, idea=description)
        finally:
            await handle.terminate_now()

        console.print(
            Panel(
                summary,
                title="training with the idea below",
                box=rich.box.DOUBLE,
            )
        )
        score, run_metrics = _get_score(absolute_wt_script_path, reference=reference)

    return ExperimentResult(branch=branch, score=score, run_metrics=run_metrics)


def append_record(results_path: Path, record: ExperimentRecord) -> None:
    """Append one experiment record to the results log as a line of JSON.

    JSONL (append-per-iteration) rather than a single JSON dump at the end so a
    crash or interrupt mid-loop keeps every experiment already completed — each
    ``train_model`` run can take hours, so partial results are worth preserving.
    """
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a") as f:
        f.write(record.model_dump_json() + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str, help="Path to the script to be improved.")
    parser.add_argument(
        "--max-num-ideas", type=int, default=20, help="Maximum number of ideas to try."
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="autoresearch_results.jsonl",
        help="Path to the JSONL file where experiment records are appended.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    repo_root = _repo_root(Path(args.script))
    base_branch = _current_branch(repo_root)

    # Resolve to an absolute path so the log lands where the user expects
    # regardless of any CWD changes (e.g. git-worktree work) later on.
    results_path = Path(args.results_path).absolute()
    # Start each run clean: drop records from any previous run.
    results_path.unlink(missing_ok=True)
    logger.info(f"writing experiment records to {results_path}")

    tried_ideas: list[ResearchIdea] = []

    # Generate baseline results.
    console.print(
        Panel(
            "",
            title="Baseline",
            box=rich.box.DOUBLE,
        )
    )

    def _run_baseline():
        baseline_score, baseline_run_metrics = _get_score(Path(args.script).absolute())
        best_score = baseline_score.mean
        # The baseline's raw per-task means become the reference every later idea
        # is normalized against, so improvements are measured relative to it.
        reference = _reference_from_score(baseline_score)
        logger.info(f"baseline score = {best_score:0.3f}; reference = {reference}")
        # Record the baseline like any other experiment so downstream tooling can
        # treat it uniformly. It has no research idea and is the accepted starting
        # point of the hill-climb, hence iteration -1 and accepted=True.
        append_record(
            results_path,
            ExperimentRecord(
                iteration=-1,
                idea=ResearchIdea(
                    summary="baseline",
                    description="The unmodified starting implementation.",
                ),
                result=ExperimentResult(
                    branch=base_branch,
                    score=baseline_score,
                    run_metrics=baseline_run_metrics,
                ),
                accepted=True,
                best_score=best_score,
            ),
        )
        return best_score, reference

    best_score, reference = _run_baseline()

    # Propose and experiment ideas.
    for iteration in range(args.max_num_ideas):
        idea: ResearchIdea = asyncio.run(
            make_reseach_idea(
                args.script,
                summary_tried_ideas=[_idea.summary for _idea in tried_ideas],
                print_src_code=False,
            )
        )

        console.print(
            Panel(
                Group(
                    Text(idea.summary, style="bold"),
                    Text(),
                    Markdown(idea.description),
                ),
                title="research idea",
                box=rich.box.DOUBLE,
            )
        )

        result = asyncio.run(
            apply_in_branch(
                args.script,
                summary=idea.summary,
                description=idea.description,
                reference=reference,
            )
        )
        console.print(
            f"\n[bold green]done.[/] Changes committed on branch [cyan]{result.branch}[/]\n"
            f"Review with: [dim]git diff {base_branch}...{result.branch}[/]"
        )

        # Greedy hill-climb: only keep an idea if it beats the best score so far.
        # Merging fast-forwards ``base_branch`` (and its working tree) to the
        # winning idea, so the next proposal and the next worktree build on the
        # accumulated code. Ideas that don't improve are left on their branch and
        # abandoned.
        accepted = result.score.mean > best_score
        if accepted:
            merge_branch(repo_root, result.branch)
            console.print(
                Panel(
                    Group(
                        Text(idea.summary, style="bold"),
                        Text(),
                        Text.from_markup(
                            f"new best score [cyan]{result.score.mean:.2f}[/] "
                            f"(beat prior best [cyan]{best_score:.2f}[/]); "
                            f"merged into [cyan]{base_branch}[/]"
                        ),
                    ),
                    title="[bold green]accepted[/]",
                    border_style="green",
                    box=rich.box.DOUBLE,
                )
            )
            best_score = result.score.mean
        else:
            console.print(
                Panel(
                    Group(
                        Text(idea.summary, style="bold"),
                        Text(),
                        Text.from_markup(
                            f"score [cyan]{result.score.mean:.2f}[/] "
                            f"did not beat best [cyan]{best_score:.2f}[/]; "
                            f"changes left on branch [cyan]{result.branch}[/]"
                        ),
                    ),
                    title="[bold red]rejected[/]",
                    border_style="red",
                    box=rich.box.DOUBLE,
                )
            )

        append_record(
            results_path,
            ExperimentRecord(
                iteration=iteration,
                idea=idea,
                result=result,
                accepted=accepted,
                best_score=best_score,
            ),
        )

        tried_ideas.append(idea)


if __name__ == "__main__":
    main()
