#!/usr/bin/env python
"""
Benchmarking harness comparing LLM bug-detection configs (e.g. a small
local model vs. a large hosted one, with vs. without NBFix's structural
context) over the llm_bench fixture set (tests/resources/llm_bench/ -
see that directory's README.md for the bug-class taxonomy and ground-truth
methodology).

Not part of the installed nbfix package - a dev-only research tool.
Requires `pip install -e ".[dev,llm]"` first.

Thin consumer of nbfix.llm's product code (context_builder.py's
context_mode/finding_types, DetectBugsEvent) rather than a separate,
hand-copied ablation implementation - so what this benchmarks is exactly
what the CLI/server extension actually run, not a parallel approximation
of it.

--eval-dir is expected to be organized one level deep by bug class
(<eval-dir>/<bug_class>/<name>.ipynb), matching tests/resources/llm_bench/'s
layout. Each notebook may have an optional
"<notebook>.ipynb.expected.json" sidecar (same {cell_id, path, errors}
shape as tests/rtests.py's .out files) to enable precision/recall/F1
scoring; without one, results are still recorded for manual review, just
unscored. llm_bench's clean fixtures ship an explicit empty `[]` sidecar
rather than no file, so a hallucinated finding on a clean notebook counts
as a false positive instead of being silently unscored.

Usage:
    python scripts/benchmark_llm.py \\
        --eval-dir tests/resources/llm_bench \\
        --config small=qwen2.5-coder:14b@http://localhost:11434/v1 \\
        --config large=gpt-4o@https://api.openai.com/v1 \\
        --context-configs none deps \\
        --output results/
"""
import argparse
import csv
import dataclasses
import json
import os
import time
from dataclasses import dataclass

from nbcore.analyses.dataleak_analysis import DATA_LEAK
from nbcore.analyses.dependency_analysis import build_fixpoint_dependency_edges
from nbcore.analyses.type_shape_analysis import (
    build_pruned_dependency_edges,
    detect_type_changes,
    label_stable_dependencies,
)
from nbcore.llm.client import LLMClient, LLMClientError
from nbcore.llm.detect_bugs_event import DetectBugsEvent
from nbcore.llm.notebook_loading import load_notebook_resilient
from nbcore.resource_utils.utils import load_notebook, read_json
from nbcore.session import AnalysisSession
from nbfix.analyses.idle_cell_analysis import IDLE
from nbfix.analyses.isolated_cell_analysis import ISOLATED
from nbfix.analyses.stale_cell_analysis import STALE
from nbfix.session_factory import new_session

# Named presets rather than a raw context_mode x finding_types cross-product
# - the benchmark's job is a small number of clean comparisons, not
# exhaustive coverage.
#
# Correction to an earlier assumption here: llm_bench's fixtures were
# designed so the *seeded bug* never trips STALE/DATA_LEAK (those need
# live run-history the static fixtures don't have), but that's not the
# same as "the four analyses never fire at all" - IsolatedCellAnalysis
# fires on the vast majority of cells in every size-tier fixture (85-98%
# at mini through xlarge), because the distractor/filler cells are
# deliberately self-contained and therefore genuinely isolated by that
# analysis's own definition. So "deps+findings" (and "deps+isolated"
# below) are real, non-trivial conditions on these fixtures - mostly
# real-but-irrelevant-to-the-bug findings, which is exactly the "does
# irrelevant real structure hurt" question, not a no-op.
#
# "deps+types" is a separate case again: it doesn't come from a registered
# NBFix.all_analyses entry at all, so it can't be expressed as a
# finding_types allowlist over nbfix.results. It's the standalone
# type_shape_analysis.py prototype (see that file's docstring) fed in via
# DetectBugsEvent's extra_findings escape hatch, computed fresh per run
# in run_once() below rather than via nbfix.add_analyses/run_analyses.
#
# "deps-pruned" and "deps-labeled" are the two alternatives to
# "deps+types" tested after finding 8 showed *adding* a type-change fact
# can backfire (it shifts where the model anchors its answer, away from
# the benchmark's crash-site ground truth). Instead of adding a new fact,
# both act on the existing dependency graph, using the same
# type/shape-stability walk (type_shape_analysis.py's
# _compute_type_stability) to tell confirmed-stable (boring, unlikely to
# be a type/shape-change bug) edges apart from everything else:
#   - "deps-pruned" removes confirmed-stable edges outright (via
#     build_pruned_dependency_edges, passed through DetectBugsEvent's
#     dependency_edges override).
#   - "deps-labeled" keeps the full graph but annotates each
#     confirmed-stable edge as low-risk (via label_stable_dependencies,
#     fed in through extra_findings like "deps+types").
#
# There used to be a separate "deps-fixpoint" config here, and
# "deps-labeled" used to need an explicit dependency_edges override to
# get the fixpoint graph. Both are gone now: context_builder.py's
# build_dependency_edges *is* build_fixpoint_dependency_edges (a plain
# alias - the old ID-ordered single-pass version was simply wrong, see
# that module's comment and experiments.md findings 10-11), so every
# config below that doesn't set its own dependency_edges override - i.e.
# "deps", "deps+findings", "deps+isolated", "deps+types" - already gets
# the order-independent fixpoint graph via DetectBugsEvent's normal
# default path, the same one the real CLI/server extension use.
_CONTEXT_CONFIGS = {
    "none": {"context_mode": "none", "finding_types": None},
    "deps": {"context_mode": "deps", "finding_types": None},
    "deps+findings": {"context_mode": "deps", "finding_types": {DATA_LEAK, STALE, IDLE, ISOLATED}},
    "deps+isolated": {"context_mode": "deps", "finding_types": {ISOLATED}},
    "deps+types": {"context_mode": "deps", "finding_types": None, "extra_findings": "type_shape"},
    "deps-pruned": {"context_mode": "deps", "finding_types": None, "dependency_edges": "pruned"},
    "deps-labeled": {"context_mode": "deps", "finding_types": None, "extra_findings": "stable_labels"},
    # Isolated single-variable change from "deps-labeled": identical
    # edges/labels, only label_stable_dependencies's message wording
    # shrinks (~25 words -> ~6, same facts) - see that function's
    # `terse` param docstring. Tests how sensitive the model's output is
    # to a small, same-information prompt change, not a new mechanism.
    "deps-labeled-terse": {"context_mode": "deps", "finding_types": None, "extra_findings": "stable_labels_terse"},
}


class _UsageCapturingClient:
    """
    Wraps a real LLMClient so DetectBugsEvent's call to `chat_json` (its
    only production interface) also records token usage, without
    DetectBugsEvent itself needing to know about `chat_json_with_usage` -
    that method exists on LLMClient specifically for this benchmarking
    need, per its own docstring. Keeps DetectBugsEvent's actual code path
    exercised rather than duplicating prompt-building here.
    """

    def __init__(self, inner: LLMClient):
        self._inner = inner
        self.last_usage = None

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
        findings, usage = self._inner.chat_json_with_usage(system_prompt, user_prompt)
        self.last_usage = usage
        return findings


def score_findings(
    actual: list[tuple[int, int]],
    expected: list[tuple[int, int]],
    line_tolerance: int = 2,
) -> dict:
    """
    Lenient precision/recall/F1 between two lists of (cell_id, line) pairs.

    A match requires the same cell_id and a line within line_tolerance -
    unlike rtests.py's exact-equality comparison (appropriate for
    deterministic analyses), LLM output isn't line-exact reproducible, so
    exact matching would understate quality.
    """
    remaining_actual = list(actual)
    true_positives = 0

    for exp_cell, exp_line in expected:
        match = next(
            (
                a for a in remaining_actual
                if a[0] == exp_cell and abs(a[1] - exp_line) <= line_tolerance
            ),
            None,
        )
        if match is not None:
            remaining_actual.remove(match)
            true_positives += 1

    false_positives = len(remaining_actual)
    false_negatives = len(expected) - true_positives

    precision = true_positives / len(actual) if actual else 0.0
    recall = true_positives / len(expected) if expected else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


@dataclass
class ModelConfig:
    name: str
    model: str
    base_url: str


def parse_config(spec: str) -> ModelConfig:
    name, _, rest = spec.partition("=")
    model, _, base_url = rest.partition("@")
    if not name or not model or not base_url:
        raise ValueError(f"Invalid --config {spec!r}, expected NAME=MODEL@BASE_URL")
    return ModelConfig(name=name, model=model, base_url=base_url)


@dataclass
class RunResult:
    notebook: str
    bug_class: str
    config: str
    context_config: str
    wall_clock_s: float
    tokens: dict
    finding_count: int
    raw_findings: list
    score: dict = None
    error: str = None


def _load_nbfix(notebook_json: dict, level: int = 5) -> AnalysisSession:
    """
    Builds a real NBFix session and loads the notebook via the real fail-fast core
    path first - what production (CLI/server/JupyterLab) actually runs,
    and what llm_bench's fixtures are verified to load through cleanly.
    Falls back to the LLM-only resilient loader (which skips individual
    unparseable cells rather than aborting) only if a fixture actually
    needs a construct the parser doesn't support yet - checked per
    fixture, not assumed.
    """
    nbfix = new_session(level=level)
    try:
        nbfix.load(load_notebook(notebook_json["cells"]))
    except Exception:
        nbfix.cells = load_notebook_resilient(notebook_json["cells"])
    return nbfix


def run_once(
    notebook_name: str, bug_class: str, nbfix: AnalysisSession, config: ModelConfig,
    context_config_name: str, expected,
) -> RunResult:
    context_config = _CONTEXT_CONFIGS[context_config_name]
    finding_types = context_config["finding_types"]

    if finding_types:
        # DetectBugsEvent deliberately never runs analyses itself (see its
        # docstring) - the caller populates nbfix.results first.
        nbfix.add_analyses(list(finding_types))
        nbfix.run_analyses(-1, list(finding_types))

    extra_findings_kind = context_config.get("extra_findings")
    extra_findings = None
    if extra_findings_kind:
        if extra_findings_kind == "type_shape":
            type_result = detect_type_changes(nbfix.cells)
        elif extra_findings_kind == "stable_labels_terse":
            type_result = label_stable_dependencies(nbfix.cells, terse=True)
        else:
            type_result = label_stable_dependencies(nbfix.cells)
        extra_findings = [
            error_info
            for path_result in type_result.path_results
            for error_info in path_result.error_infos
        ]

    dependency_edges_kind = context_config.get("dependency_edges")
    dependency_edges = None
    if dependency_edges_kind == "pruned":
        dependency_edges = build_pruned_dependency_edges(nbfix.cells)
    elif dependency_edges_kind == "fixpoint":
        dependency_edges = build_fixpoint_dependency_edges(nbfix.cells)

    inner_client = LLMClient(
        base_url=config.base_url,
        model=config.model,
        api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
    )
    client = _UsageCapturingClient(inner_client)
    event = DetectBugsEvent(
        "full", client=client,
        context_mode=context_config["context_mode"], finding_types=finding_types,
        extra_findings=extra_findings, dependency_edges=dependency_edges,
    )

    start = time.monotonic()
    try:
        result = nbfix.execute_event(event)
    except LLMClientError as exc:
        return RunResult(
            notebook=notebook_name, bug_class=bug_class, config=config.name,
            context_config=context_config_name, wall_clock_s=time.monotonic() - start,
            tokens={}, finding_count=0, raw_findings=[], score=None, error=str(exc),
        )

    elapsed = time.monotonic() - start
    actual_pairs = [
        (path_result.path[-1], error_info.line)
        for path_result in result.path_results
        for error_info in path_result.error_infos
    ]

    score = None
    if expected is not None:
        expected_pairs = [
            (entry["cell_id"], err["line"]) for entry in expected for err in entry["errors"]
        ]
        score = score_findings(actual_pairs, expected_pairs)

    raw_findings = json.loads(result.dumps(True)) if result.path_results else []

    return RunResult(
        notebook=notebook_name, bug_class=bug_class, config=config.name,
        context_config=context_config_name, wall_clock_s=elapsed,
        tokens=client.last_usage or {}, finding_count=len(actual_pairs),
        raw_findings=raw_findings, score=score, error=None,
    )


def _discover_eval_notebooks(eval_dir: str):
    """Yields (bug_class, notebook_filename) - one level deep,
    <eval_dir>/<bug_class>/<name>.ipynb, matching llm_bench/'s layout."""
    for bug_class in sorted(os.listdir(eval_dir)):
        class_dir = os.path.join(eval_dir, bug_class)
        if not os.path.isdir(class_dir):
            continue
        for name in sorted(os.listdir(class_dir)):
            if name.endswith(".ipynb"):
                yield bug_class, name


def _load_expected(eval_dir: str, bug_class: str, notebook_name: str):
    path = os.path.join(eval_dir, bug_class, notebook_name + ".expected.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


_CSV_FIELDS = [
    "notebook", "bug_class", "config", "context_config", "wall_clock_s",
    "prompt_tokens", "completion_tokens", "total_tokens", "finding_count",
    "true_positives", "false_positives", "false_negatives",
    "precision", "recall", "f1", "error",
]


def _write_csv(results: list, csv_path: str) -> None:
    """
    Flat, one-row-per-run CSV alongside results.jsonl - so charts (or any
    other analysis) can be regenerated later from saved data without
    rerunning the actual LLM calls. jsonl stays the full-fidelity format
    (keeps raw_findings, nested tokens/score dicts); this is a deliberately
    flattened, spreadsheet/pandas-friendly view of the same runs.
    """
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in results:
            score = r.score or {}
            writer.writerow({
                "notebook": r.notebook,
                "bug_class": r.bug_class,
                "config": r.config,
                "context_config": r.context_config,
                "wall_clock_s": r.wall_clock_s,
                "prompt_tokens": r.tokens.get("prompt_tokens"),
                "completion_tokens": r.tokens.get("completion_tokens"),
                "total_tokens": r.tokens.get("total_tokens"),
                "finding_count": r.finding_count,
                "true_positives": score.get("true_positives"),
                "false_positives": score.get("false_positives"),
                "false_negatives": score.get("false_negatives"),
                "precision": score.get("precision"),
                "recall": score.get("recall"),
                "f1": score.get("f1"),
                "error": r.error,
            })


def _print_summary(results: list) -> None:
    groups: dict = {}
    for r in results:
        groups.setdefault((r.config, r.bug_class, r.context_config), []).append(r)

    print("\n=== Summary ===")
    print(
        f"{'config':<10}{'bug_class':<22}{'context':<16}{'runs':<6}"
        f"{'avg_lat_s':<11}{'avg_tokens':<12}{'avg_prec':<10}{'avg_rec':<10}{'avg_f1':<8}"
    )
    for (config_name, bug_class, context_config), rs in sorted(groups.items()):
        ok = [r for r in rs if r.error is None]
        avg_latency = sum(r.wall_clock_s for r in ok) / len(ok) if ok else 0.0
        avg_tokens = sum(r.tokens.get("total_tokens") or 0 for r in ok) / len(ok) if ok else 0.0
        scored = [r for r in ok if r.score is not None]
        avg_precision = sum(r.score["precision"] for r in scored) / len(scored) if scored else float("nan")
        avg_recall = sum(r.score["recall"] for r in scored) / len(scored) if scored else float("nan")
        avg_f1 = sum(r.score["f1"] for r in scored) / len(scored) if scored else float("nan")
        print(
            f"{config_name:<10}{bug_class:<22}{context_config:<16}{len(rs):<6}"
            f"{avg_latency:<11.2f}{avg_tokens:<12.1f}{avg_precision:<10.2f}"
            f"{avg_recall:<10.2f}{avg_f1:<8.2f}"
        )


def _warm_up(config: ModelConfig) -> None:
    """
    Ollama evicts/reloads a model when a different one is requested -
    without this, the first real timed call to a model pays a cold-load
    cost that has nothing to do with context size. Found via the first
    llm_bench run: "none" was consistently much slower than "deps" for
    every model/class, purely because the loop called it first for that
    model each time. One throwaway call before timing starts absorbs that
    cost instead of letting it land on whichever context config runs first.
    """
    client = LLMClient(
        base_url=config.base_url, model=config.model,
        api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
    )
    try:
        client.chat_json("You are a helpful assistant.", 'Reply with {"ok": true} and nothing else.')
    except LLMClientError:
        pass  # if the model's genuinely unreachable, the real calls below will fail too and report it


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM bug-detection configs.")
    parser.add_argument("--eval-dir", required=True, help="Directory of <bug_class>/<name>.ipynb notebooks.")
    parser.add_argument(
        "--config", action="append", required=True, dest="configs",
        help="NAME=MODEL@BASE_URL, repeatable.",
    )
    parser.add_argument(
        "--context-configs", nargs="+", choices=list(_CONTEXT_CONFIGS), default=["deps"],
        help="Named context configs to run each notebook/config through.",
    )
    parser.add_argument("--output", default="benchmark_results", help="Output directory.")
    args = parser.parse_args()

    configs = [parse_config(c) for c in args.configs]
    os.makedirs(args.output, exist_ok=True)
    results_path = os.path.join(args.output, "results.jsonl")

    notebooks = []
    for bug_class, notebook_name in _discover_eval_notebooks(args.eval_dir):
        notebook_path = os.path.join(args.eval_dir, bug_class, notebook_name)
        try:
            notebook_json = read_json(notebook_path)
        except Exception as exc:
            print(f"Skipping {bug_class}/{notebook_name}: failed to read ({exc})")
            continue
        expected = _load_expected(args.eval_dir, bug_class, notebook_name)
        notebooks.append((bug_class, notebook_name, notebook_json, expected))

    all_results = []
    with open(results_path, "w") as out_f:
        # One full pass per model (not per notebook) - keeps each model
        # warm across all its runs instead of swap-thrashing between 3
        # different models every notebook, which is what produced the
        # cold-load latency confound in the first run.
        for config in configs:
            print(f"Warming up {config.name} ({config.model})...")
            _warm_up(config)

            for bug_class, notebook_name, notebook_json, expected in notebooks:
                for context_config_name in args.context_configs:
                    print(f"Running {bug_class}/{notebook_name} / {config.name} / {context_config_name}...")
                    # Fresh NBFix per run - finding_types mutates
                    # active_analyses/results, and reloading is cheap for
                    # these small fixtures, so there's no reason to risk
                    # state leaking across context configs.
                    nbfix = _load_nbfix(notebook_json)
                    result = run_once(notebook_name, bug_class, nbfix, config, context_config_name, expected)
                    if result.error:
                        print(f"  error: {result.error}")
                    all_results.append(result)
                    out_f.write(json.dumps(dataclasses.asdict(result)) + "\n")
                    out_f.flush()

    csv_path = os.path.join(args.output, "results.csv")
    _write_csv(all_results, csv_path)

    _print_summary(all_results)
    print(f"\nFull results written to {results_path}")
    print(f"Flat CSV written to {csv_path}")


if __name__ == "__main__":
    main()
