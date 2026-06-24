# AGENTS.md

Purpose
- Give succinct, actionable guidance for AI coding agents working on this repo.

Quick setup
- Install dependencies: `pip install -r requirements.txt`
- Download dataset: place `GlobalWeatherRepository.csv` in `outputs/` (see [README.md](README.md) for the Kaggle link)

Run
- Run the full pipeline: `python src/analysis.py`
- Outputs: figures in `figures/`, cleaned CSV and metrics in `outputs/`

Key files
- `src/analysis.py` — full end-to-end pipeline; primary entrypoint for runs
- `requirements.txt` — Python deps
- `outputs/GlobalWeatherRepository.csv` — required dataset (not checked in)

Agent conventions (minimal)
- Link, don't duplicate: refer to [README.md](README.md) for project context and dataset instructions.
- Avoid committing large data outputs or figures. If needed, ask before adding big binary files.
- Run `src/analysis.py` locally when proposing changes that affect outputs/figures to validate results.
- For performance-heavy changes (long model retrains), propose code-first and test on a small subset.

When to update this file
- Add short notes here when repo conventions change or when new automated hooks are added.

Contact
- No dedicated maintainer listed in repo; rely on in-repo README and commit history for authorship context.
