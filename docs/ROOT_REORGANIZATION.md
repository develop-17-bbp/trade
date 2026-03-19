# Root Reorganization Guide

This repo was reorganized to keep the project root minimal and focused on runnable code.

## Root now keeps only

- Core project files (`README.md`, `requirements.txt`, configs, env templates)
- Source and runtime directories (`src`, `tests`, `data`, `models`, `logs`, etc.)
- A small number of direct-run test scripts

## Where files were moved

- `docs/guides/`  
  Setup guides, quick starts, architecture notes, and operational playbooks.

- `docs/status/`  
  Implementation status notes, execution summaries, and point-in-time progress docs.

- `docs/reports/`  
  Historical reports, comprehensive logs, and legacy deliverable artifacts.

- `scripts/maintenance/`  
  Utility and maintenance Python scripts previously in root.

- `scripts/windows/`  
  Windows `.bat` helper scripts.

- `artifacts/logs/`  
  Root-level log and output text files that are not active source code.

## Notes

- File contents were preserved; only locations changed.
- If a command or doc references old root paths, update paths to the new folders above.
