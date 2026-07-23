"""Script editing planner — given a Python script and a task, produce an edit plan and apply it.

Usage:
    python agent.py --script sac.py
"""

import argparse
import asyncio
import subprocess
import textwrap
import uuid
from pathlib import Path

import rich.box
from ai_functions import ai_function
from ai_functions.ai_thread import PostConditionResult
from ai_functions.ai_thread.config import ThreadKwargs
from ai_functions.types.events import MessageAssistantTokenEvent, ToolCallEvent
from botocore.config import Config as BotocoreConfig
from loguru import logger
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from strands import tool
from strands.models.bedrock import BedrockModel

# isort: off
# Local sibling modules — keep grouped and in this order (ruff must not resort).
from post_conditions import train_model
from worktree import git_worktree

# isort: on

_MODEL = BedrockModel(
    model_id="us.anthropic.claude-opus-4-8",
    max_tokens=128_000,
    boto_client_config=BotocoreConfig(read_timeout=600),
)

PERFORMANCE_MEASURE = textwrap.dedent("""
    ## How is performance measured?
    - The performance of the implementation on each task is measred by the cumulative return on the task.
    - The overall performance of the implementation is measured by the average performance across all the tasks considered.
""")


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


@tool
def read_file(path: str) -> str:
    """Read and return the contents of a file at the given path."""
    return Path(path).read_text()


@ai_function(coordinator_tools_enabled=False, tools=[read_file], model=_MODEL)
def propose_idea(script_path: str) -> str:
    return textwrap.dedent(f"""
    You are an expert in reinforcement learning.
    Your job is to propose algorithmic ideas to improve the performance of a given implementation of a reinforcement learning algorithm.

    The implementation can be found here: {script_path}

    {PERFORMANCE_MEASURE}

    ## Requirement
    - Please provide exactly one idea in the form of a string with markdown syntax.
    - The idea needs to be concrete and actionable, that is, one can follow this idea to implement the algorithm easily.
    - You should **not** change the high-level algorithm. For example, if the original implementation is a soft actor-critic (SAC) algorithm, you cannot change it to Proximal Policy Optimization (PPO).

    """)  # type: ignore


def _dummy(abs_script_path: Path) -> PostConditionResult | None:
    logger.info("!!!!checking dummy post condition for debugging purposes!!!")
    PostConditionResult(passed=True, message=f"Great!!! {abs_script_path=}")


@ai_function(coordinator_tools_enabled=False)
def experiment_idea(script_path: str, idea: str) -> str:
    return textwrap.dedent(f"""
    You are an expert Python developer and reinforcement learning engineer. Implement and apply the research idea to the script to improve the algorithm's performance.

    {PERFORMANCE_MEASURE}

    Script path (relative to the worktree path): {script_path}

    Research idea: {idea}

    Steps:
    1. Use read_file to read the current script content.
    2. Use write_file to write the fully updated file after applying all edits.
    3. Call commit_changes with a concise commit message summarising the changes.

    Return a message confirming what has been done.
    """)


console = Console()


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


async def make_edit_plan(script_path: str) -> str:
    script_content = Path(script_path).read_text()
    console.print(
        Panel(
            Syntax(script_content, "python", line_numbers=True, theme="default"),
            title=f"script: {script_path}",
            box=rich.box.DOUBLE,
        )
    )

    handle = await propose_idea.spawn()
    try:
        with handle.coordinator.on(_on_event, thread_id=handle.id):
            return await handle.run(script_path=script_path)
    finally:
        await handle.terminate_now()


async def apply_in_branch(script_path: str, idea: str) -> str:
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
                    lambda _: train_model(absolute_wt_script_path),
                ],
            }

        handle = await experiment_idea.spawn(
            config_hook=config_hook,
        )
        try:
            with handle.coordinator.on(_on_event, thread_id=handle.id):
                await handle.run(script_path=wt_script, idea=idea)
        finally:
            await handle.terminate_now()

        # console.print("\n[bold yellow]running postcondition checks...[/]")
        # ok, msg = train_model(wt_path, rel_path)
        # style = "bold green" if ok else "bold red"
        # console.print(f"[{style}]postcondition:[/] {msg}")
        # if not ok:
        #     raise RuntimeError(f"Postcondition failed: {msg}")

    return branch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str, help="Path to the script to be improved.")
    return parser.parse_args()


def main():
    args = parse_args()
    idea = asyncio.run(make_edit_plan(args.script))

    console.print(
        Panel(
            Syntax(idea, "markdown", line_numbers=True, theme="default"),
            title="research idea",
            box=rich.box.DOUBLE,
        )
    )

    branch = asyncio.run(apply_in_branch(args.script, idea))
    console.print(
        f"\n[bold green]done.[/] Changes committed on branch [cyan]{branch}[/]\n"
        f"Review with: [dim]git diff main...{branch}[/]"
    )


if __name__ == "__main__":
    main()
