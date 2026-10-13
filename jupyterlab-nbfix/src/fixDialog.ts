import { Dialog, Notification, showDialog } from '@jupyterlab/apputils';
import { Widget } from '@lumino/widgets';

import { INBFixFixResult } from './handler';
import { IFixProposal, NotebookSessionManager } from './manager';

const inProgress = new WeakSet<NotebookSessionManager>();

/**
 * The "Suggest Fix" flow, shared by the command and the "Suggest fix"
 * button on data-leak diagnostics: ask the server for a repair, show it,
 * and apply it only if the user accepts. Nothing is written otherwise.
 */
export async function suggestFix(manager: NotebookSessionManager): Promise<void> {
  if (inProgress.has(manager)) {
    Notification.info('NBFix: a fix is already being prepared for this notebook.', { autoClose: 4000 });
    return;
  }
  inProgress.add(manager);
  let proposal: IFixProposal;
  try {
    const pending = manager.suggestFix();
    // Notification.promise needs a JSON-valued promise; the flow awaits the real one.
    Notification.promise(pending.then(() => null), {
      pending: { message: 'NBFix: asking the LLM for a fix - this can take a minute…' },
      success: { message: () => 'NBFix: fix ready for review.', options: { autoClose: 3000 } },
      error: { message: (reason: unknown) => `NBFix: fix failed - ${reason}` }
    });
    proposal = await pending;
  } catch {
    return; // reported by the notification above
  } finally {
    inProgress.delete(manager);
  }

  const { result } = proposal;
  if (result.status === 'no_findings') {
    Notification.info('NBFix: no data leaks to fix.', { autoClose: 4000 });
    return;
  }
  if (result.status === 'failed') {
    Notification.error(
      `NBFix: the LLM did not produce a usable fix after ${result.attempts} attempt(s): ` +
        (result.problems ?? []).join(' '),
      { autoClose: false }
    );
    return;
  }

  const verified = result.status === 'verified';
  const choice = await showDialog({
    title: verified ? 'NBFix: Suggested fix (verified)' : 'NBFix: Suggested fix (NOT verified)',
    body: new FixDialogBody(result),
    buttons: [
      Dialog.cancelButton(),
      Dialog.okButton({
        label: verified ? 'Apply' : 'Apply anyway',
        displayType: verified ? 'default' : 'warn'
      })
    ]
  });
  if (!choice.button.accept) {
    return;
  }
  if (manager.applyFix(proposal)) {
    // Data-leak diagnostics reflect what has run, so they come back once the
    // edited cells are re-run - running code is left to the user. Cells are
    // numbered from 0, as in NBFix's diagnostics and the diff.
    const first = Math.min(...(result.edits ?? []).map(edit => edit.cell));
    Notification.success(
      `NBFix: fix applied. Re-run the notebook from cell ${first} to re-check it for data leaks. ` +
        'Undo with Ctrl+Z in each edited cell.',
      { autoClose: 10000 }
    );
  } else {
    Notification.warning(
      'NBFix: the notebook changed while the fix was being prepared, so it was not applied. ' +
        'Run Suggest Fix again.',
      { autoClose: false }
    );
  }
}

/** Dialog body: verification status, the LLM's explanation, and the diff. */
class FixDialogBody extends Widget {
  constructor(result: INBFixFixResult) {
    super({ node: FixDialogBody._createNode(result) });
    this.addClass('jp-nbfix-fix-dialog');
  }

  private static _createNode(result: INBFixFixResult): HTMLElement {
    const container = document.createElement('div');
    const leaks = result.findings.length;

    const status = document.createElement('p');
    if (result.status === 'verified') {
      status.textContent =
        `NBFix re-analyzed the patched notebook: the ${leaks} data leak(s) are gone, and the ` +
        'change keeps the evaluation and preprocessing steps. Review the change before applying it - ' +
        'NBFix checks for leaks, not that the code still does what you intended.';
    } else {
      status.className = 'jp-nbfix-fix-warning';
      status.textContent = `NBFix could not verify this fix (after ${result.attempts} attempts):`;
    }
    container.appendChild(status);

    if (result.status !== 'verified' && result.problems?.length) {
      const problems = document.createElement('ul');
      for (const problem of result.problems) {
        const item = document.createElement('li');
        item.textContent = problem;
        problems.appendChild(item);
      }
      container.appendChild(problems);
    }

    if (result.explanation) {
      const explanation = document.createElement('p');
      explanation.textContent = `LLM: ${result.explanation}`;
      container.appendChild(explanation);
    }

    const diff = document.createElement('pre');
    diff.className = 'jp-nbfix-diff';
    for (const line of (result.diff ?? '').split('\n')) {
      const span = document.createElement('span');
      span.textContent = line + '\n';
      if (line.startsWith('+++') || line.startsWith('---')) {
        span.className = 'jp-nbfix-diff-file';
      } else if (line.startsWith('@@')) {
        span.className = 'jp-nbfix-diff-hunk';
      } else if (line.startsWith('+')) {
        span.className = 'jp-nbfix-diff-add';
      } else if (line.startsWith('-')) {
        span.className = 'jp-nbfix-diff-del';
      }
      diff.appendChild(span);
    }
    container.appendChild(diff);
    return container;
  }
}
