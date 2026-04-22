"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { startBot, getSession } from "@/lib/api";

type PageState = "idle" | "joining" | "active" | "error";

export default function HomePage() {
  const router = useRouter();
  const [pageState, setPageState] = useState<PageState>("idle");
  const [meetingUrl, setMeetingUrl] = useState("");
  const [changeCount, setChangeCount] = useState(0);
  const [errorMessage, setErrorMessage] = useState("");
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Poll session status every 5s when active
  useEffect(() => {
    if (pageState === "active") {
      pollIntervalRef.current = setInterval(async () => {
        try {
          const session = await getSession();
          setChangeCount(session.change_count);
          if (session.status === "ended") {
            clearInterval(pollIntervalRef.current!);
            router.push("/results");
          }
        } catch {
          // Polling errors are non-fatal; keep retrying
        }
      }, 5000);
    }

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [pageState, router]);

  async function handleJoin() {
    if (!meetingUrl.trim()) return;

    setPageState("joining");
    setErrorMessage("");

    try {
      await startBot(meetingUrl.trim());
      setPageState("active");
    } catch (err) {
      setErrorMessage(
        err instanceof Error ? err.message : "Failed to start bot"
      );
      setPageState("error");
    }
  }

  function handleRetry() {
    setPageState("idle");
    setErrorMessage("");
  }

  return (
    <main className="min-h-screen flex items-center justify-center px-4">
      <div className="w-full max-w-md">
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-8">
          {/* Header */}
          <div className="mb-8 text-center">
            <h1 className="text-2xl font-semibold text-gray-900">Jarvis</h1>
            <p className="mt-1 text-sm text-gray-500">
              Meeting assistant with Confluence review
            </p>
          </div>

          {/* Idle state: meeting URL input */}
          {pageState === "idle" && (
            <div className="space-y-4">
              <div>
                <label
                  htmlFor="meeting_url"
                  className="block text-sm font-medium text-gray-700 mb-1"
                >
                  Meeting URL
                </label>
                <input
                  id="meeting_url"
                  type="url"
                  placeholder="https://meet.google.com/abc-def-ghi"
                  value={meetingUrl}
                  onChange={(e) => setMeetingUrl(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleJoin()}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
              <button
                onClick={handleJoin}
                disabled={!meetingUrl.trim()}
                className="w-full py-2.5 px-4 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                Join Meeting
              </button>
            </div>
          )}

          {/* Joining state: spinner */}
          {pageState === "joining" && (
            <div className="flex flex-col items-center gap-4 py-4">
              <div className="w-8 h-8 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
              <p className="text-sm text-gray-600">Starting bot...</p>
            </div>
          )}

          {/* Active state: bot is in meeting */}
          {pageState === "active" && (
            <div className="space-y-6">
              <div className="flex items-center gap-3 p-4 bg-green-50 border border-green-200 rounded-lg">
                <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span className="text-sm font-medium text-green-800">
                  Bot is in the meeting
                </span>
              </div>

              <div className="text-center">
                <p className="text-3xl font-bold text-gray-900">{changeCount}</p>
                <p className="text-sm text-gray-500 mt-1">
                  Confluence {changeCount === 1 ? "change" : "changes"} queued
                </p>
              </div>

              <div className="text-center">
                <p className="text-xs text-gray-400 mb-3">
                  Automatically redirecting when meeting ends...
                </p>
                <button
                  onClick={() => router.push("/results")}
                  className="text-sm text-blue-600 underline hover:text-blue-800"
                >
                  Meeting ended? View Results
                </button>
              </div>
            </div>
          )}

          {/* Error state */}
          {pageState === "error" && (
            <div className="space-y-4">
              <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-sm text-red-700">{errorMessage}</p>
              </div>
              <button
                onClick={handleRetry}
                className="w-full py-2.5 px-4 bg-gray-100 text-gray-700 text-sm font-medium rounded-lg hover:bg-gray-200 transition-colors"
              >
                Try Again
              </button>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
