# Running NBTooling

NBTooling is two tools sharing one core library (`nbcore`):

- **NBFix** (`nbfix`) - static analysis for notebooks and scripts.
  Stale/idle/isolated cells and data leakage, plus optional LLM-assisted
  bug, stale-cell and API-call-sequence detection.
- **NBCompile** - notebook → script converter. Not built yet.

This document covers setup and commands for NBFix, as a CLI and through
a JupyterLab extension.

## Setup

```bash
git clone git@github.com:Formal-Labs-Code/NBTooling.git
cd NBTooling
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# nbcore is a dependency of both tools but isn't published anywhere, so
# install it explicitly alongside whichever tool(s) you need.
pip install -e "./nbcore[dev]" -e "./nbfix[dev]" -e "./nbcompile[dev]"
```

Install just one tool (plus `nbcore`) if you only need it, e.g.
`pip install -e "./nbcore[dev]" -e "./nbfix[dev]"` for NBFix alone.

## 1. NBFix CLI

Batch/one-shot mode - the same analyses the live JupyterLab extension
(section 2) runs continuously, run once against a notebook or script.

```bash
nbfix -f <path-to-notebook-or-script> -a <analysis-name> [<analysis-name> ...] [-s <start-cell>] [-l <level>]
```

| Flag | Meaning |
| --- | --- |
| `-f`, `--filename` | Path to a `.ipynb` notebook or `.py` script. Mutually exclusive with `-n`. |
| `-n`, `--notebook` | Currently broken - see note below. Use `-f` instead. |
| `-a`, `--analyses` | One or more analysis names (space-separated), see table below. Required unless using `--detect-bugs`/`--detect-stale-cells`. |
| `-s`, `--start` | Cell id to start the analysis from. Default `0`. |
| `-l`, `--level` | Analysis depth. Default `5`. |

| Analysis name (pass exactly as shown) |
| --- |
| `Data Leak Analysis` |
| `Stale Cells Analysis` |
| `Idle Cells Analysis` |
| `Isolated Cells Analysis` |

### Examples

```bash
nbfix -f tests/resources/Basic.ipynb -a "Idle Cells Analysis"
nbfix -f tests/resources/dataleak_true.ipynb -a "Data Leak Analysis" "Stale Cells Analysis"
```

Output is a JSON array of per-cell results printed to stdout, e.g.:

```json
[{"cell_id":0,"errors":[{"line":1,"label":"x","error_type":"ErrorType.TERMINAL","message":"Variable is not used outside this cell."}],"path":[0]}]
```

**Known issue:** `-n`/`--notebook` is meant to accept a raw notebook JSON
string on the command line, but the CLI passes that string straight to
`load_notebook()` without parsing it, so it always fails with
`TypeError: string indices must be integers, not 'str'`. Use `-f` with a
file path instead.

### LLM-assisted detection (optional, `nbfix[llm]`)

```bash
nbfix -f tests/resources/Basic.ipynb --detect-bugs --scope full
nbfix -f tests/resources/Basic.ipynb --detect-stale-cells --cell 0
```

Points at a local Ollama instance by default; override with the
`NBCORE_LLM_BASE_URL`/`NBCORE_LLM_MODEL` env vars, plus
`NBCORE_LLM_API_KEY` for a hosted OpenAI-compatible endpoint (see
nbcore's `llm/config.py`).

### Fixing data leaks (optional, `nbfix[llm]`)

```bash
nbfix -f tests/resources/dataleak_script_flat_true.py --fix
nbfix -f tests/resources/dataleak_script_flat_true.py --fix --diff | git apply
```

`--fix` asks the LLM for a change that removes every data leak, then
re-runs the analysis on the patched code. The repair is `verified` only
if no leaks remain, it adds no new analysis warnings, and every call a
leak was reported at (`predict`, `score`, ...) is still there. A rejected
attempt is retried with the reasons as feedback (`--attempts`, default 3).

The file is never written. `--fix` prints JSON with a `status`
(`verified`, `unverified`, `failed` or `no_findings`), the proposed
`edits` (whole replacement cells), a unified `diff` and the LLM's
`explanation`; an `unverified` result also lists the `problems` and
`remaining_findings`. `--fix --diff` prints only the diff of a verified
repair - for a script, a diff of the file that `git apply` accepts; for a
notebook, one diff per edited cell to read, not apply.

### Scripts

A `.py` script with `# %%` (VS Code / Jupytext) or `# In[ ]:` (Jupyter
export) cell markers is split into one cell per marker and analyzed like
a notebook. A script without markers is a single cell, so only
`Data Leak Analysis` applies to it - the stale/idle/isolated analyses are
about notebook execution state.

```bash
nbfix -f <path-to-script.py> -a "Data Leak Analysis" [-s <start>] [-l <level>]
nbfix -f <path-to-script.py> --detect-bugs --scope full
```

## 2. JupyterLab extension

The fully working, live-diagnostics integration: squiggly underlines on
notebook cells as you edit, add, remove, and run them. Split into two
packages so installing the JupyterLab UI (which needs `jupyterlab` and a
Node/npm toolchain) never weighs down a plain `pip install nbfix`:

- `nbfix[jupyter]` - the `jupyter_server` REST extension
  (`nbfix/src/nbfix/serverextension/`) that runs NBFix's
  analysis engine and returns diagnostics over HTTP. Pure Python, no
  Node required.
- `jupyterlab-nbfix/` - the JupyterLab labextension (TypeScript) that
  calls that API and renders diagnostics via CodeMirror. Building it
  requires Node/npm.

### Setup

```bash
git clone git@github.com:Formal-Labs-Code/NBTooling.git
cd NBTooling
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 1. Install nbcore + NBFix with the server extension
pip install -e "./nbcore" -e "./nbfix[jupyter]"

# 2. Build and install the labextension (requires Node.js/npm)
pip install ./jupyterlab-nbfix
```

The second `pip install` runs `npm install` and a webpack build under the
hood via `hatch-jupyter-builder` - expect it to take a minute or two the
first time.

### Verify both pieces registered

```bash
jupyter server extension list
# should show: nbfix.serverextension  enabled  OK

jupyter labextension list
# should show: jupyterlab-nbfix v0.1.0  enabled  OK  (python, nbfix)
```

### Run it

```bash
jupyter lab
```

Open any notebook. The extension opens a session for it automatically
(`open_notebook` event) and starts reporting diagnostics as you run cells
and edit/add/remove them; no further setup or commands are needed once
JupyterLab is running.

### Suggesting a fix for a data leak

Hover a data-leak warning and click **Suggest fix** in its tooltip, or run
**NBFix: Suggest Fix for Data Leaks** from the command palette. NBFix asks
the LLM (see "Fixing data leaks" above; it needs `nbfix[llm]` installed
next to the server extension) for a repair of every data leak in the
notebook as it is now, verifies it, and shows a dialog with the
verification status, the LLM's explanation and the diff. This can take a
minute with a local model.

Nothing changes until you click **Apply** (**Apply anyway** for a fix NBFix
could not verify). Apply is refused if the notebook changed while the fix
was being prepared. After applying, re-run the notebook from the first
edited cell: data-leak warnings reflect what has run. Each edited cell can
be undone with Ctrl+Z.

### Rebuilding after frontend changes

If you edit `jupyterlab-nbfix/src/*.ts`, rebuild with:

```bash
cd jupyterlab-nbfix
npm run build:prod
```

then restart `jupyter lab` (a hard browser refresh may also be needed to
pick up the new bundle).

### Or: run it in Docker (no local Node/Python setup needed)

`Dockerfile` builds the same server extension + labextension above into a
self-contained image, so you don't need Node, npm, or a Python venv on your
own machine at all:

```bash
docker build -t nbfix-jupyterlab .
docker run -p 8888:8888 -v "$(pwd):/home/nbfix/work" nbfix-jupyterlab
```

JupyterLab generates a fresh access token on every start and prints it to
the container logs - no token is baked into the image. Check
`docker logs <container>` (or the terminal, if run in the foreground) for a
line like:

```
http://127.0.0.1:8888/lab?token=<token>
```

and open that URL in a browser. The `-v` mount above puts your local
directory at `/home/nbfix/work` inside the container, so notebooks
you open/save there persist on your host machine; drop it if you just
want to try NBFix against the container's own filesystem.

The build is a multi-stage Dockerfile - Node.js is only used in the build
stage to compile the labextension and never ends up in the final image, so
the runtime image is close to a plain JupyterLab install rather than
carrying a full Node toolchain.
