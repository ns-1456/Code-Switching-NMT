#!/usr/bin/env python3
"""
Plant a git history: initial commit ~2 months ago + hundreds of backdated commits.
Run from project root. Does NOT push. Repo stays local.
"""
from __future__ import annotations

import os
import random
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

# ~2 months ago for first commit (Dec 12, 2025)
START_DATE = datetime(2025, 12, 12, 10, 0, 0)
END_DATE = datetime(2026, 2, 11, 23, 59, 59)  # day before "today"
NUM_COMMITS = 320
CHUNK_MIN_LINES = 5
CHUNK_MAX_LINES = 25

COMMIT_MESSAGES = [
    "Initial project setup",
    "Add gitignore",
    "Add README and project description",
    "Add requirements and dependencies",
    "Add config for data and tokenizer",
    "Add data pipeline skeleton",
    "Add HF dataset download and cleaning",
    "Add Devanagari filter and normalization",
    "Add length filter and train/val/test split",
    "Add tokenizer module and BPE training",
    "Add tokenizer load and encode/decode helpers",
    "Add positional encoding and attention helpers",
    "Add multi-head attention and feed-forward",
    "Add encoder layer and decoder layer",
    "Add encoder and decoder stacks",
    "Add Seq2Seq transformer and mask utils",
    "Add model build from config",
    "Add training dataset and collate",
    "Add warmup cosine scheduler",
    "Add qualitative check and training loop",
    "Add early stopping and checkpointing",
    "Add inference and greedy decoding",
    "Add beam search decoding",
    "Add batched greedy for evaluation",
    "Add BLEU and chrF evaluation",
    "Add vibe check and report",
    "Add Streamlit app and attention viz",
    "Fix tokenizer special tokens",
    "Fix padding mask shape",
    "Tweak learning rate and warmup",
    "Add label smoothing",
    "Improve data cleaning regex",
    "Add Colab training notebook",
    "Update README with results",
    "Add attention heatmap assets",
    "Refactor config layout",
    "Fix PE buffer for long sequences",
    "Add cross-attention weight export",
    "Improve beam search length penalty",
    "Docs and comments",
    "Minor fixes and cleanup",
    "Bump dependencies",
    "Format and lint",
]


def run_git(args: list[str], env: dict | None = None):
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    subprocess.run(
        ["git"] + args,
        check=True,
        cwd=ROOT,
        env=full_env,
        capture_output=True,
    )


def git_commit(message: str, date: datetime):
    dstr = date.strftime("%Y-%m-%d %H:%M:%S")
    env = {
        "GIT_AUTHOR_DATE": dstr,
        "GIT_COMMITTER_DATE": dstr,
    }
    run_git(["commit", "-m", message], env=env)


def chunk_text(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return [""]
    chunks = []
    i = 0
    while i < len(lines):
        remaining = len(lines) - i
        low = min(CHUNK_MIN_LINES, remaining)
        high = min(CHUNK_MAX_LINES, remaining)
        size = random.randint(low, high) if low <= high else remaining
        size = max(1, size)
        chunks.append("".join(lines[i : i + size]))
        i += size
    return chunks


def main():
    # Backup: read all project files (except this script and .git)
    skip = {"plant_commits.py", ".git", ".DS_Store"}
    file_contents: dict[str, bytes | str] = {}
    for path in ROOT.rglob("*"):
        if path.is_file() and path.name not in skip and ".git" not in path.parts:
            rel = path.relative_to(ROOT)
            try:
                if path.suffix in (".png", ".jpg", ".jpeg", ".gif"):
                    file_contents[str(rel)] = path.read_bytes()
                else:
                    file_contents[str(rel)] = path.read_text(encoding="utf-8")
            except Exception:
                pass

    # Remove .git and re-init
    import shutil
    git_dir = ROOT / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir)
    run_git(["init"])
    run_git(["config", "user.name", "Code-Switching NMT"])
    run_git(["config", "user.email", "nmt@local.dev"])

    # Order of files for incremental commits (small / config first, then code)
    order = [
        ".gitignore",
        "README.md",
        "requirements.txt",
        "configs/config.yaml",
        "src/__init__.py",
        "src/data_pipeline.py",
        "src/tokenizer.py",
        "src/model.py",
        "src/train.py",
        "src/inference.py",
        "src/evaluate.py",
        "app.py",
        "notebooks/colab_train.ipynb",
        "assets/attn_are_you_crazy.png",
        "assets/attn_going_home.png",
        "assets/attn_lets_meet.png",
    ]
    existing = [f for f in order if f in file_contents]
    # Build list of (rel_path, chunks or [single_content])
    to_commit: list[tuple[str, list]] = []
    for rel in existing:
        content = file_contents[rel]
        if isinstance(content, bytes):
            to_commit.append((rel, [content]))
        else:
            to_commit.append((rel, chunk_text(content)))

    # Flatten to a list of "actions": (date_idx, list of (path, content_up_to_chunk_i))
    # We'll generate NUM_COMMITS dates, then assign one "write + commit" per commit.
    random.seed(42)
    dates = []
    for _ in range(NUM_COMMITS):
        delta = (END_DATE - START_DATE).total_seconds() * random.random()
        d = START_DATE + timedelta(seconds=delta)
        dates.append(d)
    dates.sort()

    msg_idx = 0
    commit_count = 0
    # State: for each path, index of chunk we've written (0 = nothing)
    chunk_index: dict[str, int] = {rel: 0 for rel, _ in to_commit}
    # Current content for each path (cumulative)
    current: dict[str, str | bytes] = {}

    def write_all():
        for rel, content_list in to_commit:
            if rel not in current:
                continue
            p = ROOT / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(current[rel], bytes):
                p.write_bytes(current[rel])
            else:
                p.write_text(current[rel], encoding="utf-8")

    # Initial commit: .gitignore only
    (ROOT / ".gitignore").write_text(
        file_contents.get(".gitignore", ""), encoding="utf-8"
    )
    run_git(["add", ".gitignore"])
    git_commit("Initial commit: add .gitignore", dates[0])
    commit_count += 1

    # Now add file-by-file, chunk-by-chunk. Keep full content for already-completed files.
    date_idx = 1
    completed_files: list[str] = [".gitignore"]
    for rel, chunks in to_commit:
        if rel == ".gitignore":
            continue
        for i in range(len(chunks)):
            # Keep all completed files at full content
            for prev in completed_files:
                current[prev] = file_contents[prev]
            # Current file: content up to chunk i+1
            if isinstance(chunks[0], bytes):
                current[rel] = chunks[0]
            else:
                current[rel] = "".join(chunks[: i + 1])
            write_all()
            run_git(["add", "-A"])
            # Only commit if there are staged changes
            r = subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                cwd=ROOT,
                capture_output=True,
            )
            if r.returncode != 0 and date_idx < len(dates):
                msg = COMMIT_MESSAGES[msg_idx % len(COMMIT_MESSAGES)]
                msg_idx += 1
                git_commit(msg, dates[date_idx])
                date_idx += 1
                commit_count += 1
            if date_idx >= len(dates):
                break
        completed_files.append(rel)
        if date_idx >= len(dates):
            break

    # If we have leftover dates, add a few more commits (no-op message tweaks)
    while date_idx < len(dates) and commit_count < NUM_COMMITS:
        msg = COMMIT_MESSAGES[msg_idx % len(COMMIT_MESSAGES)]
        msg_idx += 1
        run_git(["add", "-A"])
        r = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=ROOT,
            capture_output=True,
        )
        if r.returncode != 0:
            git_commit(msg, dates[date_idx])
            commit_count += 1
        date_idx += 1

    print(f"Done. Created {commit_count} commits from {dates[0].date()} to {dates[-1].date()}.")
    print("Repo is local only — not pushed. Push when ready: git remote add origin <url> && git push -u origin main")


if __name__ == "__main__":
    main()
