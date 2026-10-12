# NBHarness

Live, real-time notebook diagnostics - stale/idle/isolated cells and
data leakage, plus LLM-assisted stale-cell and API-call-sequence
detection. Flags problems as you edit and run cells; never repairs.

Built on `nbcore`. Ships a `jupyter_server` REST extension
(`serverextension/`) for live diagnostics in JupyterLab (see the
`jupyterlab-nbharness/` labextension), and its own `nbharness` CLI for
batch/one-shot scanning of a notebook file outside a live session.

See the top-level repo README for how this fits into the wider suite.

## MCP wrapper (PoC 0)

The NBHarness package can also expose the analysis through a very thin MCP
server. The wrapper calls the existing `nbharness.cli.nbharness()` batch
analysis entrypoint and translates its raw JSON output into the PoC contract.

Install the MCP extra:

```bash
cd NBTooling
. .venv/bin/activate
python -m pip install -e './nbharness[mcp]'
```

Start the server over stdio:

```bash
python -m nbharness.mcp_server
```

The server exposes the primary tool `analyze`. The `verify` helper remains as
an optional, best-effort runtime sanity check and is not the core NBHarness
contract.

The `analyze` tool accepts a notebook (`.ipynb` path, or a notebook
dict/JSON payload) or a Python script (`.py` path, raw source, or
`{"script": "<source>"}`), and returns one finding per raw analyzer error.

## Python scripts

Both the CLI (`nbharness -f script.py -a "Data Leak Analysis"`) and the
MCP `analyze` tool accept `.py` files:

- A script with cell markers - `# %%` (VS Code / Jupytext; `# %% [markdown]`
  sections are skipped) or `# In[ ]:` (Jupyter's `.py` export) - is split
  into one cell per marker and analyzed exactly like a notebook.
- A script without markers is a single cell, analyzed top to bottom. Only
  the data-leak analysis applies; the stale/idle/isolated analyses are about
  notebook execution state and are rejected for such scripts.

Script results add `file_line` (the line in the `.py` file) next to each
cell-relative `line`, for the sink and for the finding's source.

Example direct smoke-check:

```bash
python scripts/nbharness_mcp_smoke.py
```
