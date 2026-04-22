export type BotStatus = "idle" | "joining" | "in_meeting" | "ended" | "error";

export interface SessionStatus {
  status: BotStatus;
  bot_id: string | null;
  meeting_url: string | null;
  change_count: number;
  error?: string;
}

export type ChangeType = "create" | "edit" | "delete" | "title";

export interface ChangeItem {
  id: number;
  change_type: ChangeType;
  page_id: string | null;
  page_title: string;
  section_heading: string | null;
  before_content: string | null;
  after_content: string | null;
  timestamp: string;
  session_id: string;
  status: "pending" | "approved" | "rejected" | "executed" | "failed";
}

export interface ChangeResult {
  id: number;
  success: boolean;
  error?: string;
}

export interface ExecuteChangesResponse {
  results: ChangeResult[];
}

export interface MeetingSummary {
  title: string;
  date: string;
  summary: string;
  key_topics: string[];
  action_items: ActionItem[];
  decisions: string[];
  participants: string[];
}

export interface ActionItem {
  description: string;
  owner?: string;
}
