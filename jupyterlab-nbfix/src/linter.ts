import { Diagnostic, linter } from '@codemirror/lint';
import { StateEffect, StateField } from '@codemirror/state';
import { EditorView } from '@codemirror/view';
import {
  CodeMirrorEditor,
  EditorExtensionRegistry,
  IEditorExtensionRegistry
} from '@jupyterlab/codemirror';

import { INBFixDiagnostic } from './handler';

const invalidate = StateEffect.define<null>();

const invalidationCounter = StateField.define<number>({
  create: () => 0,
  update: (value, tr) => {
    for (const effect of tr.effects) {
      if (effect.is(invalidate)) {
        value += 1;
      }
    }
    return value;
  }
});

/**
 * Diagnostics are pushed here per editor view; the CodeMirror linter reads
 * from this store (a pull-based source, per CodeMirror 6's design) rather
 * than diagnostics being dispatched directly into editor state.
 */
class DiagnosticsStore {
  private _diagnostics = new WeakMap<EditorView, Diagnostic[]>();

  set(view: EditorView, diagnostics: Diagnostic[]): void {
    this._diagnostics.set(view, diagnostics);
    view.dispatch({ effects: invalidate.of(null) });
  }

  clear(view: EditorView): void {
    this.set(view, []);
  }

  get(view: EditorView): Diagnostic[] {
    return this._diagnostics.get(view) ?? [];
  }
}

export const diagnosticsStore = new DiagnosticsStore();

/**
 * Converts one cell's NBFix error list into CodeMirror Diagnostic ranges,
 * locating the label text on its reported line when present (mirrors the
 * VS Code prototype's diagnostic-range logic in server_handler.ts).
 */
export function toCodeMirrorDiagnostics(
  view: EditorView,
  cellDiagnostic: INBFixDiagnostic
): Diagnostic[] {
  const diagnostics: Diagnostic[] = [];
  const lineCount = view.state.doc.lines;

  for (const error of cellDiagnostic.errors) {
    const lineNo = Math.min(Math.max(error.line, 1), lineCount);
    const line = view.state.doc.line(lineNo);
    let from = line.from;
    let to = line.to;

    if (error.label) {
      const index = line.text.indexOf(error.label);
      if (index >= 0) {
        from = line.from + index;
        to = from + error.label.length;
      }
    }

    const isLLMFinding = error.error_type.startsWith('LLM_');
    // Deterministic analyses (STALE/IDLE/ISOLATED/DATA_LEAK) all render as
    // 'warning' today regardless of their own error_type - unchanged here,
    // out of scope for this pass. Only LLM-sourced findings get a distinct
    // severity/markClass so they're visually distinguishable as
    // AI-suggested rather than statically proven.
    const severity = isLLMFinding
      ? error.error_type === 'LLM_CRITICAL'
        ? 'error'
        : 'warning'
      : 'warning';

    diagnostics.push({
      from,
      to,
      severity,
      message: error.message,
      source: 'nbfix',
      ...(isLLMFinding ? { markClass: 'cm-nbfix-llm-finding' } : {})
    });
  }

  return diagnostics;
}

/**
 * Registers the NBFix linter as a global editor extension so any code
 * editor JupyterLab creates (including every notebook cell) can display
 * diagnostics pushed via diagnosticsStore.set().
 */
export function registerNBFixLinter(registry: IEditorExtensionRegistry): void {
  registry.addExtension({
    name: 'nbfix:diagnostics',
    factory: () => {
      const source = (view: EditorView): Diagnostic[] => diagnosticsStore.get(view);

      const nbfixLinter = linter(source, {
        delay: 0,
        needsRefresh: update => {
          const previous = update.startState.field(invalidationCounter);
          const current = update.state.field(invalidationCounter);
          return previous !== current;
        }
      });

      return EditorExtensionRegistry.createImmutableExtension([
        nbfixLinter,
        invalidationCounter
      ]);
    }
  });
}

export function getEditorView(editor: unknown): EditorView | undefined {
  return (editor as CodeMirrorEditor | undefined)?.editor;
}
