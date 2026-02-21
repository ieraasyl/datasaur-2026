import { useState } from "react";
import type { DiagnoseResponse } from "./types";

interface Props {
  onResult: (data: DiagnoseResponse) => void;
  onLoading: (loading: boolean) => void;
  onError: (err: string | null) => void;
}

export default function SymptomForm({ onResult, onLoading, onError }: Props) {
  const [symptoms, setSymptoms] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!symptoms.trim() || loading) return;

    setLoading(true);
    onLoading(true);
    onError(null);

    try {
      const res = await fetch("/diagnose", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symptoms }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: DiagnoseResponse = await res.json();
      onResult(data);
    } catch (e: unknown) {
      onError(e instanceof Error ? e.message : "Ошибка запроса");
    } finally {
      setLoading(false);
      onLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      handleSubmit();
    }
  };

  return (
    <div className="w-full">
      <div
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: "12px",
          overflow: "hidden",
          transition: "border-color 0.2s",
        }}
        onFocus={() => {}}
      >
        <textarea
          value={symptoms}
          onChange={(e) => setSymptoms(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Опишите симптомы пациента на русском или английском языке...&#10;&#10;Например: Пациент жалуется на боль в грудной клетке, одышку при физической нагрузке, отёки нижних конечностей. АД 160/100. Длительность симптомов — 2 недели."
          disabled={loading}
          style={{
            width: "100%",
            minHeight: "180px",
            padding: "20px",
            background: "transparent",
            border: "none",
            outline: "none",
            color: "var(--text)",
            fontFamily: "'Manrope', sans-serif",
            fontSize: "15px",
            lineHeight: "1.6",
            resize: "vertical",
          }}
        />
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            padding: "12px 20px",
            borderTop: "1px solid var(--border)",
          }}
        >
          <span style={{ color: "var(--text-muted)", fontSize: "12px", fontFamily: "'DM Mono', monospace" }}>
            Ctrl+Enter для отправки
          </span>
          <button
            onClick={handleSubmit}
            disabled={loading || !symptoms.trim()}
            style={{
              padding: "10px 24px",
              background: loading || !symptoms.trim() ? "var(--accent-dim)" : "var(--accent)",
              color: "#0a0f0d",
              border: "none",
              borderRadius: "8px",
              fontFamily: "'Manrope', sans-serif",
              fontWeight: 600,
              fontSize: "14px",
              cursor: loading || !symptoms.trim() ? "not-allowed" : "pointer",
              transition: "all 0.15s",
              opacity: loading || !symptoms.trim() ? 0.6 : 1,
            }}
          >
            {loading ? "Анализ..." : "Определить диагноз →"}
          </button>
        </div>
      </div>
    </div>
  );
}
