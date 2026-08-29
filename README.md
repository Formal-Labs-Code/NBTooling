<p align="center">
  <img src="assets/nbtooling-logo.jpeg" alt="NBTooling" width="500">
</p>

# NBTooling

[![Tests](https://github.com/Formal-Labs-Code/NBTooling/actions/workflows/tests.yml/badge.svg)](https://github.com/Formal-Labs-Code/NBTooling/actions/workflows/tests.yml)

Static analysis for data science notebooks and scripts - a shared core
library plus three tools, each with one job.

## The tools

- **[NBHarness](nbharness/)** - live, real-time notebook diagnostics.
  Stale/idle/isolated cells and data leakage, plus optional LLM-assisted
  stale-cell and API-call-sequence detection. Flags problems as you edit
  and run cells; never repairs.
- **[NBFix](nbfix/)** - batch analysis and repair for scripts. Data
  leakage plus optional LLM-assisted bug detection. The only tool that
  repairs; does not handle notebooks.
- **[NBCompile](nbcompile/)** - notebook → script converter. Uses the
  real dependency graph to compute true execution order, not a naive
  cell-order dump. Not implemented yet - currently a package stub.

All three are built on **[nbcore](nbcore/)**: the parser, CFG builder,
def-use analysis, dependency graph, and analysis framework they share.
It's a library, not a tool - nothing to run directly.

## Getting started

See [`RUNNING.md`](RUNNING.md) for setup and usage of each tool, as a
CLI and (for NBHarness) through a JupyterLab extension.

Quick version:

```bash
git clone git@github.com:Formal-Labs-Code/NBTooling.git
cd NBTooling
python3 -m venv .venv
source .venv/bin/activate
pip install -e "./nbcore[dev]" -e "./nbharness[dev]" -e "./nbfix[dev]"
pytest
```

## Project layout

```
nbcore/       shared parser/CFG/dependency-graph/analysis framework
nbharness/    live notebook diagnostics (CLI + JupyterLab server extension)
nbfix/        batch script analysis and repair (CLI)
nbcompile/    notebook -> script converter (stub, not implemented)
jupyterlab-nbharness/   JupyterLab labextension (TypeScript) for NBHarness
```

Each tool directory has its own README with more detail.
