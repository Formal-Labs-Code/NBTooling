import { URLExt } from '@jupyterlab/coreutils';
import { ServerConnection } from '@jupyterlab/services';

/**
 * Call the NBFix server extension API.
 *
 * @param endPoint API REST end point for the extension
 * @param init Initial values for the request
 * @returns The response body interpreted as JSON
 */
export async function requestAPI<T>(
  endPoint = '',
  init: RequestInit = {}
): Promise<T> {
  const settings = ServerConnection.makeSettings();
  const requestUrl = URLExt.join(settings.baseUrl, 'nbfix', 'api', endPoint);

  let response: Response;
  try {
    response = await ServerConnection.makeRequest(requestUrl, init, settings);
  } catch (error) {
    throw new ServerConnection.NetworkError(error as any);
  }

  let data: any = await response.text();

  if (data.length > 0) {
    try {
      data = JSON.parse(data);
    } catch (error) {
      console.log('Not a JSON response body.', response);
    }
  }

  if (!response.ok) {
    throw new ServerConnection.ResponseError(response, data.message || data);
  }

  return data;
}

export interface INBFixDiagnostic {
  cell_id: number;
  errors: Array<{
    line: number;
    label: string;
    error_type: string;
    message: string;
  }>;
}

export interface INBFixEventResponse {
  status: 'success' | 'error';
  diagnostics?: INBFixDiagnostic[];
  // Only present for detect_api_sequence_llm - which cell_ids the check
  // actually considered, so a narrowed (focus_cell) scan's caller knows
  // which prior findings are safe to discard vs untouched. See
  // DetectApiSequenceEvent's docstring.
  checked_cells?: number[];
  message?: string;
}

/**
 * Sends a single NBFix event to the server extension's event endpoint.
 */
export function postEvent(
  eventName: string,
  notebookId: string,
  params?: Record<string, unknown>
): Promise<INBFixEventResponse> {
  return requestAPI<INBFixEventResponse>('event', {
    method: 'POST',
    body: JSON.stringify({
      event: eventName,
      notebook_id: notebookId,
      params: params ?? {}
    })
  });
}
