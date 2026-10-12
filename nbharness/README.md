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

The `analyze` tool accepts either a notebook path or a raw notebook
dict/JSON payload and returns one finding per raw analyzer error.

Example direct smoke-check:

```bash
python scripts/nbharness_mcp_smoke.py
```
