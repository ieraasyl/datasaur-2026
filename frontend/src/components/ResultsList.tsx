import DiagnosisCard from "./DiagnosisCard";
import type { DiagnoseResponse } from "./types";

interface Props {
  data: DiagnoseResponse;
}

export default function ResultsList({ data }: Props) {
  if (!data.diagnoses.length) {
    return (
      <p style={{ color: "var(--text-muted)", fontStyle: "italic" }}>
        Диагнозы не определены.
      </p>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "4px" }}>
        <span style={{ color: "var(--text-muted)", fontSize: "12px", fontFamily: "'DM Mono', monospace", textTransform: "uppercase", letterSpacing: "0.1em" }}>
          Результаты — {data.diagnoses.length} гипотез
        </span>
        <div style={{ flex: 1, height: "1px", background: "var(--border)" }} />
      </div>
      {data.diagnoses.map((d, i) => (
        <DiagnosisCard key={d.icd10_code + i} diagnosis={d} delay={i * 80} />
      ))}
    </div>
  );
}
