import type { Diagnosis } from "./types";

interface Props {
  diagnosis: Diagnosis;
  delay?: number;
}

const rankColors = ["#2dff7a", "#7affb8", "#b8ffd4"];

export default function DiagnosisCard({ diagnosis, delay = 0 }: Props) {
  const color = rankColors[diagnosis.rank - 1] ?? "#2dff7a";

  return (
    <div
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: "12px",
        padding: "20px 24px",
        animation: `slideIn 0.4s ease forwards`,
        animationDelay: `${delay}ms`,
        opacity: 0,
      }}
    >
      <style>{`
        @keyframes slideIn {
          from { opacity: 0; transform: translateY(12px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>

      <div style={{ display: "flex", alignItems: "flex-start", gap: "16px" }}>
        {/* Rank badge */}
        <div
          style={{
            minWidth: "36px",
            height: "36px",
            borderRadius: "8px",
            background: `${color}18`,
            border: `1px solid ${color}40`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontFamily: "'DM Mono', monospace",
            fontSize: "13px",
            fontWeight: 500,
            color: color,
          }}
        >
          #{diagnosis.rank}
        </div>

        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap", marginBottom: "6px" }}>
            <span style={{ fontFamily: "'Instrument Serif', serif", fontSize: "18px", color: "var(--text)" }}>
              {diagnosis.diagnosis}
            </span>
            <span
              style={{
                fontFamily: "'DM Mono', monospace",
                fontSize: "11px",
                padding: "3px 8px",
                background: `${color}15`,
                border: `1px solid ${color}35`,
                borderRadius: "4px",
                color: color,
                letterSpacing: "0.05em",
              }}
            >
              {diagnosis.icd10_code}
            </span>
          </div>

          <p
            style={{
              margin: 0,
              fontSize: "14px",
              lineHeight: "1.6",
              color: "var(--text-muted)",
            }}
          >
            {diagnosis.explanation}
          </p>
        </div>
      </div>
    </div>
  );
}
