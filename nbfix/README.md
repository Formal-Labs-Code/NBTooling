# NBFix

Static analysis for data science notebooks and scripts - stale/idle/isolated
cells and data leakage, plus LLM-assisted bug, stale-cell and
API-call-sequence detection.

Built on `nbcore`. Ships a `jupyter_server` REST extension
(`serverextension/`) for live diagnostics in JupyterLab (see the
`jupyterlab-nbfix/` labextension), an `nbfix` CLI for batch/one-shot
scanning of a notebook or script, and an MCP server.

See the top-level repo README for how this fits into the wider suite.

## MCP wrapper (PoC 0)

The NBFix package can also expose the analysis through a very thin MCP
server. The wrapper calls the existing `nbfix.cli.nbfix()` batch
analysis entrypoint and translates its raw JSON output into the PoC contract.

Install the MCP extra:

```bash
cd NBTooling
. .venv/bin/activate
python -m pip install -e './nbfix[mcp]'
```

Start the server over stdio:

```bash
python -m nbfix.mcp_server
```

The server exposes two tools, `analyze` and `fix`. The `verify` helper remains
as an optional, best-effort runtime sanity check and is not the core NBFix
contract.

The `analyze` tool accepts a notebook (`.ipynb` path, or a notebook
dict/JSON payload) or a Python script (`.py` path, raw source, or
`{"script": "<source>"}`), and returns `{"findings": [...], "warnings": [...]}`:
one finding per raw analyzer error, and one warning per piece of code the
analysis could only partly model - a statement using an unsupported construct
(analyzed as a no-op) or a local call it couldn't inline (analyzed as an
opaque call). Each warning has `cell`, `line`, `message` and, for scripts,
`file_line`. No findings for code with warnings is not a guarantee it has no
leak.

The `fix` tool takes the same input and suggests a repair for every data leak
(it needs the `llm` extra too: `pip install -e './nbfix[mcp,llm]'`). An LLM -
a local Ollama by default, configured with `NBCORE_LLM_BASE_URL`,
`NBCORE_LLM_MODEL` and, for a hosted endpoint, `NBCORE_LLM_API_KEY` (set them
in the MCP server's `env`) - proposes whole-cell replacements, and NBFix
re-analyzes the patched code. The result is `verified` only if no leaks remain,
the patch adds no new analysis warnings, and every call a leak was reported at
is still there; rejected attempts are retried with feedback (`max_attempts`,
default 3). Nothing is written: the result carries the `status`, the proposed
`edits` (`[{"cell", "code"}]`), a unified `diff` (of the script file, for
script input) and the LLM's `explanation`, for the caller to review and apply.

## Python scripts

Both the CLI (`nbfix -f script.py -a "Data Leak Analysis"`) and the
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
python scripts/nbfix_mcp_smoke.py
```
