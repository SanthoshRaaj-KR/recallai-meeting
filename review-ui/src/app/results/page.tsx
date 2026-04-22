"use client";

import { useState, useEffect } from "react";
import { getMeetingSummary, getChanges, executeChanges } from "@/lib/api";
import type { MeetingSummary, ChangeItem, ChangeResult } from "@/types";

type ExecutionState = "idle" | "executing" | "done";

export default function ResultsPage() {
  const [summary, setSummary] = useState<MeetingSummary | null>(null);
  const [changes, setChanges] = useState<ChangeItem[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [loadingError, setLoadingError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [executionState, setExecutionState] = useState<ExecutionState>("idle");
  const [results, setResults] = useState<Map<number, ChangeResult>>(new Map());

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      setLoadingError(null);
      try {
        const [summaryData, changesData] = await Promise.all([
          getMeetingSummary(),
          getChanges(),
        ]);
        setSummary(summaryData);
        setChanges(changesData);
        // Select all pending changes by default
        const pendingIds = new Set(
          changesData
            .filter((c) => c.status === "pending")
            .map((c) => c.id)
        );
        setSelectedIds(pendingIds);
      } catch (err) {
        setLoadingError(
          err instanceof Error ? err.message : "Failed to load meeting data"
        );
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  function toggleChange(id: number) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }

  async function handleApprove() {
    if (selectedIds.size === 0) return;
    setExecutionState("executing");
    try {
      const response = await executeChanges(Array.from(selectedIds));
      const resultMap = new Map<number, ChangeResult>();
      for (const r of response.results) {
        resultMap.set(r.id, r);
      }
      setResults(resultMap);
      setExecutionState("done");
    } catch (err) {
      setLoadingError(
        err instanceof Error ? err.message : "Failed to execute changes"
      );
      setExecutionState("idle");
    }
  }

  function getChangeBadgeColor(changeType: ChangeItem["change_type"]) {
    switch (changeType) {
      case "create":
        return "bg-green-100 text-green-800";
      case "edit":
        return "bg-blue-100 text-blue-800";
      case "delete":
        return "bg-red-100 text-red-800";
      case "title":
        return "bg-yellow-100 text-yellow-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  }

  if (loading) {
    return (
      <main className="min-h-screen flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-8 h-8 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
          <p className="text-sm text-gray-600">Loading meeting results...</p>
        </div>
      </main>
    );
  }

  if (loadingError) {
    return (
      <main className="min-h-screen flex items-center justify-center px-4">
        <div className="max-w-md w-full p-6 bg-red-50 border border-red-200 rounded-xl">
          <p className="text-sm text-red-700">{loadingError}</p>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-gray-50 py-10 px-4">
      <div className="max-w-3xl mx-auto space-y-8">
        {/* Page heading */}
        <div>
          <h1 className="text-2xl font-semibold text-gray-900">
            Meeting Results
          </h1>
          <p className="mt-1 text-sm text-gray-500">
            Review the meeting summary and approve Confluence changes below.
          </p>
        </div>

        {/* Section 1: Meeting Summary */}
        {summary && (
          <section className="bg-white rounded-2xl border border-gray-200 shadow-sm p-6 space-y-5">
            <div className="flex items-start justify-between">
              <div>
                <h2 className="text-lg font-semibold text-gray-900">
                  {summary.title || "Meeting Summary"}
                </h2>
                {summary.date && (
                  <p className="text-xs text-gray-400 mt-0.5">{summary.date}</p>
                )}
              </div>
            </div>

            {summary.summary && (
              <p className="text-sm text-gray-700 leading-relaxed">
                {summary.summary}
              </p>
            )}

            {summary.key_topics?.length > 0 && (
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-2">
                  Key Topics
                </h3>
                <ul className="list-disc list-inside space-y-1">
                  {summary.key_topics.map((topic, i) => (
                    <li key={i} className="text-sm text-gray-700">
                      {topic}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {summary.action_items?.length > 0 && (
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-2">
                  Action Items
                </h3>
                <ul className="list-disc list-inside space-y-1">
                  {summary.action_items.map((item, i) => (
                    <li key={i} className="text-sm text-gray-700">
                      {item.description}
                      {item.owner && (
                        <span className="text-gray-400"> — {item.owner}</span>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {summary.decisions?.length > 0 && (
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-2">
                  Decisions Made
                </h3>
                <ul className="list-disc list-inside space-y-1">
                  {summary.decisions.map((decision, i) => (
                    <li key={i} className="text-sm text-gray-700">
                      {decision}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {summary.participants?.length > 0 && (
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-2">
                  Participants
                </h3>
                <div className="flex flex-wrap gap-2">
                  {summary.participants.map((p, i) => (
                    <span
                      key={i}
                      className="px-2.5 py-0.5 bg-gray-100 text-gray-700 text-xs rounded-full"
                    >
                      {p}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </section>
        )}

        {/* Section 2: Proposed Confluence Changes */}
        <section className="space-y-4">
          <h2 className="text-lg font-semibold text-gray-900">
            Proposed Confluence Updates ({changes.length})
          </h2>

          {changes.length === 0 ? (
            <div className="bg-white rounded-2xl border border-gray-200 p-6">
              <p className="text-sm text-gray-500 text-center">
                No Confluence changes were proposed during this meeting.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {changes.map((change) => {
                const result = results.get(change.id);
                return (
                  <div
                    key={change.id}
                    className="bg-white rounded-2xl border border-gray-200 shadow-sm p-5 space-y-4"
                  >
                    {/* Change header */}
                    <div className="flex items-start gap-3">
                      <input
                        type="checkbox"
                        id={`change-${change.id}`}
                        checked={selectedIds.has(change.id)}
                        onChange={() => toggleChange(change.id)}
                        disabled={executionState !== "idle"}
                        className="mt-0.5 h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 cursor-pointer"
                      />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <label
                            htmlFor={`change-${change.id}`}
                            className="text-sm font-medium text-gray-900 cursor-pointer"
                          >
                            {change.page_title}
                          </label>
                          <span
                            className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${getChangeBadgeColor(change.change_type)}`}
                          >
                            {change.change_type}
                          </span>
                        </div>
                        {change.section_heading && (
                          <p className="text-xs text-gray-500 mt-0.5">
                            Section: {change.section_heading}
                          </p>
                        )}
                      </div>

                      {/* Per-change result indicator */}
                      {result && (
                        <span
                          className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                            result.success
                              ? "bg-green-100 text-green-700"
                              : "bg-red-100 text-red-700"
                          }`}
                        >
                          {result.success ? "Applied" : "Failed"}
                        </span>
                      )}
                    </div>

                    {/* Diff view */}
                    {(change.before_content || change.after_content) && (
                      <div className="space-y-2">
                        {change.before_content && (
                          <div className="rounded-lg overflow-hidden">
                            <div className="px-3 py-1 bg-red-100 text-xs font-medium text-red-700">
                              Before
                            </div>
                            <pre className="px-3 py-2 bg-red-50 text-xs text-red-900 whitespace-pre-wrap break-words font-mono">
                              {change.before_content}
                            </pre>
                          </div>
                        )}
                        {change.after_content && (
                          <div className="rounded-lg overflow-hidden">
                            <div className="px-3 py-1 bg-green-100 text-xs font-medium text-green-700">
                              After
                            </div>
                            <pre className="px-3 py-2 bg-green-50 text-xs text-green-900 whitespace-pre-wrap break-words font-mono">
                              {change.after_content}
                            </pre>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Error message for failed changes */}
                    {result && !result.success && result.error && (
                      <p className="text-xs text-red-600">{result.error}</p>
                    )}
                  </div>
                );
              })}

              {/* Approve button */}
              <div className="pt-2">
                <button
                  onClick={handleApprove}
                  disabled={
                    selectedIds.size === 0 || executionState !== "idle"
                  }
                  className="w-full py-3 px-4 bg-blue-600 text-white text-sm font-medium rounded-xl hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
                >
                  {executionState === "executing" && (
                    <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  )}
                  {executionState === "executing"
                    ? "Applying..."
                    : executionState === "done"
                    ? "Changes Applied"
                    : `Approve Selected Changes (${selectedIds.size})`}
                </button>
              </div>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
