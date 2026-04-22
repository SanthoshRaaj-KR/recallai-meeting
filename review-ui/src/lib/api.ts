import type {
  SessionStatus,
  ChangeItem,
  ExecuteChangesResponse,
  MeetingSummary,
} from "@/types";

const BASE = "/api";

async function request<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json() as Promise<T>;
}

/**
 * Start the meeting bot for a given meeting URL.
 * POST /bot/start
 */
export async function startBot(meeting_url: string): Promise<SessionStatus> {
  return request<SessionStatus>("/bot/start", {
    method: "POST",
    body: JSON.stringify({ meeting_url }),
  });
}

/**
 * Get current session/bot status.
 * GET /bot/status
 */
export async function getSession(): Promise<SessionStatus> {
  return request<SessionStatus>("/bot/status");
}

/**
 * Get pending Confluence changes for review.
 * GET /review/changes
 */
export async function getChanges(): Promise<ChangeItem[]> {
  return request<ChangeItem[]>("/review/changes");
}

/**
 * Execute (apply) the selected change IDs.
 * POST /review/execute
 */
export async function executeChanges(
  ids: number[]
): Promise<ExecuteChangesResponse> {
  return request<ExecuteChangesResponse>("/review/execute", {
    method: "POST",
    body: JSON.stringify({ ids }),
  });
}

/**
 * Get the meeting summary (topics, decisions, action items, participants).
 * GET /review/summary
 */
export async function getMeetingSummary(): Promise<MeetingSummary> {
  return request<MeetingSummary>("/review/summary");
}
