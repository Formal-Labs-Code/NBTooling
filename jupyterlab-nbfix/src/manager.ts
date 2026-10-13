import { Notification } from '@jupyterlab/apputils';
import { Cell, ICellModel } from '@jupyterlab/cells';
import { IObservableList } from '@jupyterlab/observables';
import { NotebookActions, NotebookPanel } from '@jupyterlab/notebook';

import { INBFixDiagnostic, postEvent } from './handler';
import { diagnosticsStore, getEditorView, toCodeMirrorDiagnostics } from './linter';

const CHANGE_DEBOUNCE_MS = 500;
// Matches events.py's AddCellEvent `kind` values: only code cells are
// tracked in NBFix's notebook_IR, so newly added cells are always
// reported as kind 2 (code) regardless of their actual cell type.
const CODE_CELL_KIND = 2;

// Matches constants.py's exact string values - same duplicated-but-
// exact-literal precedent as CODE_CELL_KIND above. These four are real
// NBFix.all_analyses entries the backend's active_analyses list
// understands (via add_active_analyses); LLM_STALE_KEY/
// LLM_API_SEQUENCE_KEY are not - neither has a backend registry entry at
// all, they're frontend-only flags gating whether detect_stale_cells_llm/
// detect_api_sequence_llm also get called (see _onCellExecuted). All
// kinds are presented together in the same "Choose Active Analyses"
// picker (index.ts) since from the user's perspective they're all just
// "an analysis I can turn on or off", even though only the first four
// round-trip through the backend's own analysis-toggle mechanism.
export const DATA_LEAK_KEY = 'Data Leak Analysis';
export const STALE_KEY = 'Stale Cells Analysis';
export const IDLE_KEY = 'Idle Cells Analysis';
export const ISOLATED_KEY = 'Isolated Cells Analysis';
export const LLM_STALE_KEY = 'llm_stale';
export const LLM_API_SEQUENCE_KEY = 'llm_api_sequence';

const DEFAULT_ANALYSES = [DATA_LEAK_KEY, STALE_KEY, IDLE_KEY, ISOLATED_KEY];

export interface IAnalysisOption {
  key: string;
  label: string;
}

// Order matters here - it's the order the picker dialog lists them in.
export const ANALYSIS_OPTIONS: IAnalysisOption[] = [
  { key: DATA_LEAK_KEY, label: 'Data Leakage' },
  { key: STALE_KEY, label: 'Stale Cells (Algorithmic)' },
  { key: IDLE_KEY, label: 'Idle Cells' },
  { key: ISOLATED_KEY, label: 'Isolated Cells' },
  { key: LLM_STALE_KEY, label: 'Stale Cells (LLM)' },
  { key: LLM_API_SEQUENCE_KEY, label: 'API Call Sequence (LLM)' }
];

/**
 * Merges two diagnostic sets by cell_id, concatenating each cell's errors.
 * Used to combine the always-on deterministic analyses with the on-demand
 * LLM findings without either one clobbering the other when re-rendered.
 */
function mergeDiagnostics(
  a: INBFixDiagnostic[],
  b: INBFixDiagnostic[]
): INBFixDiagnostic[] {
  const byCellId = new Map<number, INBFixDiagnostic['errors']>();
  for (const source of [a, b]) {
    for (const cellDiagnostic of source) {
      const existing = byCellId.get(cellDiagnostic.cell_id) ?? [];
      byCellId.set(cellDiagnostic.cell_id, [...existing, ...cellDiagnostic.errors]);
    }
  }
  return Array.from(byCellId.entries()).map(([cell_id, errors]) => ({
    cell_id,
    errors
  }));
}

function countFindings(diagnostics?: INBFixDiagnostic[]): number {
  return (diagnostics ?? []).reduce((sum, cellDiagnostic) => sum + cellDiagnostic.errors.length, 0);
}

/**
 * Wires one NotebookPanel's lifecycle/edit/execute signals to the NBFix
 * server extension, and renders the diagnostics it returns via the
 * CodeMirror linter registered in linter.ts.
 *
 * Cell identity for the backend is purely positional (matches events.py's
 * position-based Event model), so every event resolves a cell's index from
 * its live position in `panel.content.widgets` at the moment the event
 * fires, rather than capturing an index up front - positions shift under
 * concurrent adds/removes/debounced edits otherwise.
 */
export class NotebookSessionManager {
  private _panel: NotebookPanel;
  private _notebookId: string;
  private _pendingEdits = new Map<string, number>();
  private _watched = new WeakSet<ICellModel>();
  private _deterministicDiagnostics: INBFixDiagnostic[] = [];
  private _llmDiagnostics: INBFixDiagnostic[] = [];
  private _llmStaleDiagnostics: INBFixDiagnostic[] = [];
  private _llmApiSequenceDiagnostics: INBFixDiagnostic[] = [];
  private _activeAnalyses = new Set<string>(DEFAULT_ANALYSES);
  private _llmStaleEnabled = false;
  private _llmStaleWarned = false;
  private _llmApiSequenceEnabled = false;
  private _llmApiSequenceWarned = false;
  // Incremented on every _checkApiSequence call - see that method's
  // docstring for the out-of-order-response race this guards against.
  private _apiSequenceRequestSeq = 0;
  // Per-cell "code this cell was last confirmed to have actually run
  // with" - the frontend's own copy of what the backend's
  // last_ran_code tracks, kept separately because DetectStaleCellsEvent
  // needs the pre-run value captured *before* run_cell fires (which
  // updates the backend's own last_ran_code as part of the same call -
  // see that event's docstring for why reading it server-side after the
  // fact doesn't work). Keyed by cell model id (stable across
  // reordering), not position.
  private _lastConfirmedCode = new Map<string, string>();

  constructor(panel: NotebookPanel) {
    this._panel = panel;
    this._notebookId = panel.context.path;
  }

  /** Current on/off state for every option ANALYSIS_OPTIONS lists - for populating the picker dialog. */
  get activeAnalyses(): Set<string> {
    const current = new Set(this._activeAnalyses);
    if (this._llmStaleEnabled) {
      current.add(LLM_STALE_KEY);
    }
    if (this._llmApiSequenceEnabled) {
      current.add(LLM_API_SEQUENCE_KEY);
    }
    return current;
  }

  /**
   * Applies a new selection from the "Choose Active Analyses" picker.
   * The four deterministic keys round-trip through the backend's real
   * active_analyses list (add_active_analyses fully *replaces* it, not
   * merges - the full resulting set is always sent, matching
   * AddActiveAnalysesEvent's own semantics). LLM_STALE_KEY/
   * LLM_API_SEQUENCE_KEY never reach the backend at all - they only gate
   * whether _onCellExecuted also calls detect_stale_cells_llm/
   * detect_api_sequence_llm going forward.
   */
  async setActiveAnalyses(next: Set<string>): Promise<void> {
    const deterministic = [...next].filter(
      key => key !== LLM_STALE_KEY && key !== LLM_API_SEQUENCE_KEY
    );
    this._activeAnalyses = new Set(deterministic);
    this._llmStaleEnabled = next.has(LLM_STALE_KEY);
    // Newly-on (not already on) - only scan on the on->on transition, not
    // every time the dialog is re-applied with it already enabled.
    const apiSequenceNewlyEnabled = !this._llmApiSequenceEnabled && next.has(LLM_API_SEQUENCE_KEY);
    this._llmApiSequenceEnabled = next.has(LLM_API_SEQUENCE_KEY);

    const response = await postEvent('add_active_analyses', this._notebookId, {
      active_analyses: deterministic
    });
    this._setDeterministicDiagnostics(response.diagnostics);

    if (apiSequenceNewlyEnabled) {
      // Otherwise a notebook that's already fully written - opened and
      // run without ever being edited - would never get checked at all,
      // since the check is edit-triggered (see _sendChangeCell's
      // docstring for why it's edit-, not execution-, triggered).
      // Turning the toggle on should surface what's already there
      // immediately, not only warn about future edits. No focusCell -
      // a full baseline scan of every cell.
      void this._checkApiSequence();
    }
  }

  /**
   * Checks the notebook for API call-sequence violations in a single
   * LLM call. With no focusCell, scans every cell (used when the
   * toggle is switched on - a full baseline scan is the point). With a
   * focusCell, the backend narrows the prompt to that cell's
   * dependency-connected component instead of the whole notebook (see
   * DetectApiSequenceEvent/api_sequence_context_builder.py) - much
   * cheaper on every edit than re-showing the entire notebook each
   * time, especially as notebooks grow.
   *
   * Merges by checked_cells (always returned by this event, covering
   * every cell in the no-focusCell case): drop whatever findings
   * previously existed for cells that were just re-checked, then add
   * the fresh ones. Cells outside checked_cells are left untouched -
   * critical for a focusCell scan, where most of the notebook wasn't
   * re-examined at all and any existing findings there are still valid.
   */
  private async _checkApiSequence(focusCell?: number): Promise<void> {
    // Out-of-order-response guard: running cell 7 then quickly running
    // cell 8 fires two overlapping checks - the first (slower, based on
    // only cell 7 having run) can return *after* the second (faster,
    // based on the now-correct state), silently overwriting a correct
    // result with a stale wrong one. Nothing about promise ordering
    // guarantees responses arrive in request order. Confirmed as the
    // actual live cause (not hypothetical) via a direct server-pipeline
    // reproduction: running cell 7 alone gives a wrong finding, running
    // cell 7 then cell 8 gives the correct one - the live bug was the
    // wrong one winning the race. Each call gets a ticket; a response is
    // only applied if no newer call has started since.
    const ticket = ++this._apiSequenceRequestSeq;
    try {
      const response = await postEvent('detect_api_sequence_llm', this._notebookId, {
        ...(focusCell !== undefined ? { focus_cell: focusCell } : {})
      });
      if (ticket !== this._apiSequenceRequestSeq) {
        // A newer check has started since this one was fired - this
        // response is stale, whatever it says. Drop it rather than let
        // it clobber a possibly-already-applied newer result.
        return;
      }
      const checked = new Set(response.checked_cells ?? []);
      const kept = this._llmApiSequenceDiagnostics.filter(d => !checked.has(d.cell_id));
      this._llmApiSequenceDiagnostics = [...kept, ...(response.diagnostics ?? [])];
      this._render();
    } catch (error) {
      if (ticket === this._apiSequenceRequestSeq) {
        this._warnLLMApiSequenceFailure(error);
      }
    }
  }

  async start(): Promise<void> {
    await this._panel.context.ready;

    const response = await postEvent('open_notebook', this._notebookId, {
      notebook_json: this._cellsAsJSON()
    });
    this._setDeterministicDiagnostics(response.diagnostics);

    this._panel.content.model?.cells.changed.connect(this._onCellsChanged, this);
    NotebookActions.executed.connect(this._onCellExecuted, this);
    this._panel.disposed.connect(this._onDisposed, this);

    this._panel.content.widgets.forEach(cellWidget => this._watchCell(cellWidget));
  }

  /**
   * Runs an on-demand LLM bug check over the connected-cell subgraph around
   * cellIndex (not just that one cell in isolation - cross-cell bugs are
   * the priority target). Findings are merged with, not replacing, the
   * live deterministic diagnostics already on screen. Returns the number
   * of findings - Notification.promise (the caller) requires a
   * JSON-serializable resolution value, and the count doubles as a useful
   * success message.
   */
  async checkCellForBugs(cellIndex: number): Promise<number> {
    const response = await postEvent('detect_bugs', this._notebookId, {
      scope: 'subgraph',
      cell_index: cellIndex
    });
    this._setLLMDiagnostics(response.diagnostics);
    return countFindings(response.diagnostics);
  }

  /** Runs an on-demand LLM bug check over the whole notebook. */
  async checkNotebookForBugs(): Promise<number> {
    const response = await postEvent('detect_bugs', this._notebookId, {
      scope: 'full'
    });
    this._setLLMDiagnostics(response.diagnostics);
    return countFindings(response.diagnostics);
  }

  private _cellsAsJSON(): Array<{ cell_type: string; source: string }> {
    return this._panel.content.widgets.map(cellWidget => ({
      cell_type: cellWidget.model.type,
      source: cellWidget.model.sharedModel.getSource()
    }));
  }

  private _watchCell(cellWidget: Cell): void {
    if (this._watched.has(cellWidget.model)) {
      return;
    }
    this._watched.add(cellWidget.model);
    if (cellWidget.model.type !== 'code') {
      // notebook_IR (backend) only ever contains code cells - watching a
      // markdown cell's edits and sending its position through change_cell/
      // detect_api_sequence_llm was a real, previously-crashing no-op
      // (see DetectApiSequenceEvent's focus_cell guard). Not connecting
      // the handler at all avoids the wasted round trip entirely rather
      // than relying on the backend to silently absorb it.
      return;
    }
    cellWidget.model.contentChanged.connect(() => this._scheduleChangeCell(cellWidget));
  }

  private _scheduleChangeCell(cellWidget: Cell): void {
    const model = cellWidget.model;
    const existing = this._pendingEdits.get(model.id);
    if (existing !== undefined) {
      window.clearTimeout(existing);
    }

    const handle = window.setTimeout(() => {
      this._pendingEdits.delete(model.id);
      void this._sendChangeCell(cellWidget);
    }, CHANGE_DEBOUNCE_MS);

    this._pendingEdits.set(model.id, handle);
  }

  /**
   * Posts the current in-editor source for one cell as a change_cell event
   * and applies the resulting diagnostics. Shared by the debounce timeout
   * (_scheduleChangeCell) and _flushPendingEdit (called synchronously
   * before execution instead of waiting out the debounce - see its
   * docstring for why).
   */
  private async _sendChangeCell(cellWidget: Cell): Promise<void> {
    const index = this._panel.content.widgets.indexOf(cellWidget);
    if (index === -1) {
      // Cell was removed before this fired.
      return;
    }
    const response = await postEvent('change_cell', this._notebookId, {
      new_code: cellWidget.model.sharedModel.getSource(),
      cell_index: index,
      with_result: true
    });
    // The edited cell's code changed, so any prior LLM finding about it
    // (or cells connected to it) may no longer apply - drop the whole LLM
    // bug-detection/stale-cell set rather than show a possibly-stale AI
    // finding. api-sequence is deliberately NOT cleared here - see below,
    // it invalidates itself precisely via checked_cells instead.
    this._llmDiagnostics = [];
    this._llmStaleDiagnostics = [];
    this._setDeterministicDiagnostics(response.diagnostics);

    if (this._llmApiSequenceEnabled) {
      // Deliberately triggered on EDIT, not on execution (unlike the
      // stale-cell check) - the point is to warn the user before they
      // run a cell that would violate a library's call-order contract,
      // not after the kernel has already hit the real exception. Firing
      // here means the warning is typically already visible in the
      // gutter well before Shift+Enter; _flushPendingEdit calling this
      // synchronously right before execution is a backstop for the case
      // where the user edits and immediately runs within the debounce
      // window, not the primary trigger.
      //
      // Passes this cell as focusCell - _checkApiSequence narrows the
      // LLM prompt to its dependency-connected component instead of the
      // whole notebook (much cheaper on every keystroke-triggered edit),
      // and merges by checked_cells rather than wiping the whole
      // diagnostic set the way _llmDiagnostics/_llmStaleDiagnostics do
      // above - editing cell N can invalidate or produce a finding
      // anywhere in N's component, but cells outside it are provably
      // unaffected and must keep whatever findings they already had.
      void this._checkApiSequence(index);
    }
  }

  /**
   * Cancels and immediately runs any debounced change_cell POST still
   * pending for this cell, awaiting it before returning.
   *
   * Must happen before _onCellExecuted posts run_cell. Executing a cell
   * right after typing in it (plain Shift+Enter, the normal workflow)
   * fires NotebookActions.executed almost immediately - a trivial kernel
   * assignment finishes faster than CHANGE_DEBOUNCE_MS. Without this
   * flush, run_cell can reach the backend while its notebook_IR entry
   * still holds the *pre-edit* cell_code (events.py's RunCellEvent diffs
   * cell_code against last_ran_code), so STALE analysis silently diffs
   * old-against-old and finds nothing - confirmed as the actual cause of
   * a live report ("edited cell 0, re-ran it, nothing happened"), not a
   * hypothetical race.
   */
  private async _flushPendingEdit(cellWidget: Cell): Promise<void> {
    const model = cellWidget.model;
    const pending = this._pendingEdits.get(model.id);
    if (pending === undefined) {
      return;
    }
    window.clearTimeout(pending);
    this._pendingEdits.delete(model.id);
    await this._sendChangeCell(cellWidget);
  }

  private _onCellsChanged(
    _cells: unknown,
    change: IObservableList.IChangedArgs<ICellModel>
  ): void {
    if (change.type === 'add') {
      this._llmDiagnostics = [];
      this._llmStaleDiagnostics = [];
      this._llmApiSequenceDiagnostics = [];
      void postEvent('add_cell', this._notebookId, {
        position: change.newIndex,
        kind: CODE_CELL_KIND,
        content: change.newValues[0]?.sharedModel.getSource() ?? ''
      }).then(response => this._setDeterministicDiagnostics(response.diagnostics));

      const cellWidget = this._panel.content.widgets[change.newIndex];
      if (cellWidget) {
        this._watchCell(cellWidget);
      }
    } else if (change.type === 'remove') {
      this._llmDiagnostics = [];
      this._llmStaleDiagnostics = [];
      this._llmApiSequenceDiagnostics = [];
      const removedModel = change.oldValues[0];
      if (removedModel) {
        this._lastConfirmedCode.delete(removedModel.id);
      }
      void postEvent('remove_cell', this._notebookId, {
        position: change.oldIndex
      }).then(response => this._setDeterministicDiagnostics(response.diagnostics));
    }
    // 'move'/'set' changes aren't reported as distinct events yet - cell
    // reordering falls back to whatever the next run/edit on the affected
    // cells resolves to.
  }

  private _onCellExecuted(
    _emitter: unknown,
    args: { cell: Cell; success: boolean }
  ): void {
    if (!args.success) {
      return;
    }
    void this._handleCellExecuted(args.cell);
  }

  private async _handleCellExecuted(cell: Cell): Promise<void> {
    // Make sure the backend has this cell's current code before computing
    // anything below - see _flushPendingEdit's docstring for the race this
    // closes.
    await this._flushPendingEdit(cell);

    const index = this._panel.content.widgets.indexOf(cell);
    if (index === -1) {
      return;
    }
    // Running a cell doesn't change its source, so any existing LLM bug
    // finding about that code is still valid - only the deterministic set
    // is refreshed here. Awaited (not fire-and-forget) because
    // RunCellEvent is what updates last_ran_code server-side, and the
    // api-sequence check below depends on that (not_yet_run is
    // last_ran_code != cell_code) - firing it before this lands would
    // see the just-executed cell as still not-yet-run.
    const runCellResponse = await postEvent('run_cell', this._notebookId, { cell_index: index });
    this._setDeterministicDiagnostics(runCellResponse.diagnostics);

    if (this._llmApiSequenceEnabled) {
      // Execution, not just edit, can change this check's answer: a
      // cell's not_yet_run status flips purely by being run, with no
      // text edit involved at all. Edit-only triggering missed this - a
      // notebook that's already fully written, run cell-by-cell via the
      // "play" button without ever typing anything, would never get
      // re-checked, and any earlier finding would sit there stale
      // forever. Confirmed via a live report (pressing play on cells 7
      // and 8 still showed a finding on cell 9 that the prompt fix had
      // already corrected) - the gap was this missing trigger, not the
      // prompt. Uses the executed cell as focusCell, same narrowed/
      // merged scan as the edit path.
      void this._checkApiSequence(index);
    }

    // Read the pre-run code *before* anything below updates the tracked
    // value - the LLM stale check needs the code this cell actually ran
    // with last time, not what it's about to be updated to. Tracked
    // (and kept current) regardless of whether the LLM check is enabled,
    // so turning it on later immediately has an accurate baseline
    // instead of comparing against nothing.
    const model = cell.model;
    const originalCode = this._lastConfirmedCode.get(model.id) ?? '';
    const currentCode = model.sharedModel.getSource();

    if (this._llmStaleEnabled) {
      // This fires automatically on every cell run, unlike checkCellForBugs/
      // checkNotebookForBugs (user-initiated, so a Notification.promise
      // per call is appropriate) - a failure here (llm extra not
      // installed, or its configured endpoint unreachable; both are real
      // 400/500 responses from the server, not something that should ever
      // throw uncaught - see dispatch.py's _build_detect_stale_cells_llm_event
      // and handlers.py's broad except Exception) must not raise an
      // unhandled promise rejection on every keystroke-triggered run. Warn
      // the user once per session rather than either staying silent or
      // popping a notification on every single cell execution.
      void postEvent('detect_stale_cells_llm', this._notebookId, {
        cell_index: index,
        original_code: originalCode
      })
        .then(response => this._setLLMStaleDiagnostics(response.diagnostics))
        .catch(error => this._warnLLMStaleFailure(error));
    }

    this._lastConfirmedCode.set(model.id, currentCode);
  }

  private _warnLLMStaleFailure(error: unknown): void {
    console.error('nbfix: LLM stale-cell check failed', error);
    if (this._llmStaleWarned) {
      return;
    }
    this._llmStaleWarned = true;
    Notification.warning(
      `NBFix: LLM stale-cell detection is enabled but failed (${String(error)}). ` +
        'Check that the llm extra is installed and its configured endpoint is reachable. ' +
        'Not shown again this session.',
      { autoClose: 8000 }
    );
  }

  private _warnLLMApiSequenceFailure(error: unknown): void {
    console.error('nbfix: LLM API-sequence check failed', error);
    if (this._llmApiSequenceWarned) {
      return;
    }
    this._llmApiSequenceWarned = true;
    Notification.warning(
      `NBFix: LLM API call-sequence detection is enabled but failed (${String(error)}). ` +
        'Check that the llm extra is installed and its configured endpoint is reachable. ' +
        'Not shown again this session.',
      { autoClose: 8000 }
    );
  }

  private _setDeterministicDiagnostics(diagnostics?: INBFixDiagnostic[]): void {
    if (diagnostics) {
      this._deterministicDiagnostics = diagnostics;
    }
    this._render();
  }

  private _setLLMDiagnostics(diagnostics?: INBFixDiagnostic[]): void {
    if (diagnostics) {
      this._llmDiagnostics = diagnostics;
    }
    this._render();
  }

  private _setLLMStaleDiagnostics(diagnostics?: INBFixDiagnostic[]): void {
    if (diagnostics) {
      this._llmStaleDiagnostics = diagnostics;
    }
    this._render();
  }

  private _render(): void {
    const merged = mergeDiagnostics(
      mergeDiagnostics(
        mergeDiagnostics(this._deterministicDiagnostics, this._llmDiagnostics),
        this._llmStaleDiagnostics
      ),
      this._llmApiSequenceDiagnostics
    );

    this._panel.content.widgets.forEach(cellWidget => {
      const view = getEditorView(cellWidget.editor);
      if (view) {
        diagnosticsStore.clear(view);
      }
    });

    for (const cellDiagnostic of merged) {
      const cellWidget = this._panel.content.widgets[cellDiagnostic.cell_id];
      const view = cellWidget && getEditorView(cellWidget.editor);
      if (view) {
        diagnosticsStore.set(view, toCodeMirrorDiagnostics(view, cellDiagnostic));
      }
    }
  }

  private _onDisposed(): void {
    void postEvent('close_notebook', this._notebookId);
    for (const handle of this._pendingEdits.values()) {
      window.clearTimeout(handle);
    }
    this._pendingEdits.clear();
    this._lastConfirmedCode.clear();
  }
}
