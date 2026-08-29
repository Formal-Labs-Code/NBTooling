# Running NBTooling

NBTooling is three tools sharing one core library (`nbcore`):

- **NBHarness** (`nbharness`) - live, real-time notebook diagnostics.
  Stale/idle/isolated cells and data leakage, plus optional LLM-assisted
  stale-cell and API-call-sequence detection. Flags problems; never
  repairs. Works on notebooks only.
- **NBFix** (`nbfix`) - batch analysis and repair for scripts. Data
  leakage plus optional LLM-assisted bug detection. Works on scripts
  only; does not handle notebooks.
- **NBCompile** - notebook → script converter. Not built yet.

This document covers setup and commands for both working tools, as a
CLI and (for NBHarness) through a JupyterLab extension.

## Setup

```bash
git clone git@github.com:Formal-Labs-Code/NBTooling.git
cd NBTooling
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# nbcore is a dependency of both tools but isn't published anywhere, so
# install it explicitly alongside whichever tool(s) you need.
pip install -e "./nbcore[dev]" -e "./nbharness[dev]" -e "./nbfix[dev]"
```

Install just one tool (plus `nbcore`) if you only need it, e.g.
`pip install -e "./nbcore[dev]" -e "./nbfix[dev]"` for NBFix alone.

## 1. NBHarness CLI (notebooks)

Batch/one-shot mode - the same analyses the live JupyterLab extension
(section 3) runs continuously, run once against a notebook file.

```bash
nbharness -f <path-to-notebook> -a <analysis-name> [<analysis-name> ...] [-s <start-cell>] [-l <level>]
```

| Flag | Meaning |
| --- | --- |
| `-f`, `--filename` | Path to a `.ipynb` notebook. Mutually exclusive with `-n`. |
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
nbharness -f tests/resources/Basic.ipynb -a "Idle Cells Analysis"
nbharness -f tests/resources/dataleak_true.ipynb -a "Data Leak Analysis" "Stale Cells Analysis"
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

### LLM-assisted detection (optional, `nbharness[llm]`)

```bash
nbharness -f tests/resources/Basic.ipynb --detect-bugs --scope full
nbharness -f tests/resources/Basic.ipynb --detect-stale-cells --cell 0
```

Points at a local Ollama instance by default; override with the
`NBCORE_LLM_BASE_URL`/`NBCORE_LLM_MODEL` env vars (shared with NBFix's
LLM features - see nbcore's `llm/config.py`).

## 2. NBFix CLI (scripts)

Same shape, scripts only - no `-n`, no STALE/IDLE/ISOLATED (those are
notebook-edit-lifecycle analyses with no script analog).

```bash
nbfix -f <path-to-script.py> -a "Data Leak Analysis" [-s <start>] [-l <level>]
nbfix -f <path-to-script.py> --detect-bugs --scope full
```

A script is split into pseudo-cells at top-level-statement boundaries
before analysis, so the same dependency-graph/context-building
machinery NBHarness uses for notebook cells applies here too.

## 3. JupyterLab extension (NBHarness only)

The fully working, live-diagnostics integration: squiggly underlines on
notebook cells as you edit, add, remove, and run them. Split into two
packages so installing the JupyterLab UI (which needs `jupyterlab` and a
Node/npm toolchain) never weighs down a plain `pip install nbharness`:

- `nbharness[jupyter]` - the `jupyter_server` REST extension
  (`nbharness/src/nbharness/serverextension/`) that runs NBHarness's
  analysis engine and returns diagnostics over HTTP. Pure Python, no
  Node required.
- `jupyterlab-nbharness/` - the JupyterLab labextension (TypeScript) that
  calls that API and renders diagnostics via CodeMirror. Building it
  requires Node/npm.

### Setup

```bash
git clone git@github.com:Formal-Labs-Code/NBTooling.git
cd NBTooling
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 1. Install nbcore + NBHarness with the server extension
pip install -e "./nbcore" -e "./nbharness[jupyter]"

# 2. Build and install the labextension (requires Node.js/npm)
pip install ./jupyterlab-nbharness
```

The second `pip install` runs `npm install` and a webpack build under the
hood via `hatch-jupyter-builder` - expect it to take a minute or two the
first time.

### Verify both pieces registered

```bash
jupyter server extension list
# should show: nbharness.serverextension  enabled  OK

jupyter labextension list
# should show: jupyterlab-nbharness v0.1.0  enabled  OK  (python, nbharness)
```

### Run it

```bash
jupyter lab
```

Open any notebook. The extension opens a session for it automatically
(`open_notebook` event) and starts reporting diagnostics as you run cells
and edit/add/remove them; no further setup or commands are needed once
JupyterLab is running.

### Rebuilding after frontend changes

If you edit `jupyterlab-nbharness/src/*.ts`, rebuild with:

```bash
cd jupyterlab-nbharness
npm run build:prod
```

then restart `jupyter lab` (a hard browser refresh may also be needed to
pick up the new bundle).

### Or: run it in Docker (no local Node/Python setup needed)

`Dockerfile` builds the same server extension + labextension above into a
self-contained image, so you don't need Node, npm, or a Python venv on your
own machine at all:

```bash
docker build -t nbharness-jupyterlab .
docker run -p 8888:8888 -v "$(pwd):/home/nbharness/work" nbharness-jupyterlab
```

JupyterLab generates a fresh access token on every start and prints it to
the container logs - no token is baked into the image. Check
`docker logs <container>` (or the terminal, if run in the foreground) for a
line like:

```
http://127.0.0.1:8888/lab?token=<token>
```

and open that URL in a browser. The `-v` mount above puts your local
directory at `/home/nbharness/work` inside the container, so notebooks
you open/save there persist on your host machine; drop it if you just
want to try NBHarness against the container's own filesystem.

The build is a multi-stage Dockerfile - Node.js is only used in the build
stage to compile the labextension and never ends up in the final image, so
the runtime image is close to a plain JupyterLab install rather than
carrying a full Node toolchain.
