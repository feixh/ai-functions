import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def git_worktree(branch_name: str, repo_root: Path):
    with tempfile.TemporaryDirectory() as tmp:
        wt_path = Path(tmp) / "worktree"
        subprocess.run(
            ["git", "worktree", "add", str(wt_path), "-b", branch_name],
            cwd=repo_root,
            check=True,
        )
        try:
            yield wt_path
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(wt_path)],
                cwd=repo_root,
            )
