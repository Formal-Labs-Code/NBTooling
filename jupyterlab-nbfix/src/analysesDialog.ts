import { Widget } from '@lumino/widgets';

import { IAnalysisOption } from './manager';

/**
 * Dialog body for "Choose Active Analyses" - a plain checkbox list, kept
 * in its own module (matching linter.ts/handler.ts/manager.ts's
 * one-concern-per-file split already established in this package).
 * Implements Dialog.IBodyWidget<Set<string>> (a getValue() method), so
 * a Dialog<Set<string>> built with this as its body resolves directly to
 * the selected keys on accept - no separate parsing step needed by the
 * caller.
 */
export class AnalysesDialogBody extends Widget {
  private _checkboxes = new Map<string, HTMLInputElement>();

  constructor(options: IAnalysisOption[], selected: Set<string>) {
    super({ node: AnalysesDialogBody._createNode(options, selected) });
    this.addClass('jp-nbfix-analyses-dialog');
    for (const option of options) {
      const input = this.node.querySelector<HTMLInputElement>(
        `input[data-key="${CSS.escape(option.key)}"]`
      );
      if (input) {
        this._checkboxes.set(option.key, input);
      }
    }
  }

  getValue(): Set<string> {
    const selected = new Set<string>();
    for (const [key, input] of this._checkboxes) {
      if (input.checked) {
        selected.add(key);
      }
    }
    return selected;
  }

  private static _createNode(
    options: IAnalysisOption[],
    selected: Set<string>
  ): HTMLElement {
    const container = document.createElement('div');
    for (const option of options) {
      const label = document.createElement('label');
      label.style.display = 'block';
      label.style.margin = '4px 0';

      const input = document.createElement('input');
      input.type = 'checkbox';
      input.dataset.key = option.key;
      input.checked = selected.has(option.key);
      input.style.marginRight = '6px';

      label.appendChild(input);
      label.appendChild(document.createTextNode(option.label));
      container.appendChild(label);
    }
    return container;
  }
}
