import { useState } from "react";
import SymptomForm from "./SymptomForm";
import ResultsList from "./ResultsList";
import type { DiagnoseResponse } from "./types";

export default function App() {
  const [result, setResult] = useState<DiagnoseResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  return (
    <main
      style={{
        minHeight: "100vh",
        padding: "48px 24px",
        maxWidth: "760px",
        margin: "0 auto",
      }}
    >
      {/* Header */}
      <div style={{ marginBottom: "48px" }}>
        <div style={{
          fontFamily: "'DM Mono', monospace",
          fontSize: "11px",
          color: "var(--accent)",
          letterSpacing: "0.15em",
          textTransform: "uppercase",
          marginBottom: "12px",
        }}>
          Клинические протоколы РК · МКБ-10
        </div>
        <h1 style={{
          fontFamily: "'Instrument Serif', serif",
          fontSize: "clamp(32px, 5vw, 48px)",
          fontWeight: 400,
          margin: "0 0 12px",
          lineHeight: 1.1,
          color: "var(--text)",
        }}>
          Ассистент<br />
          <em style={{ color: "var(--accent)" }}>диагностики</em>
        </h1>
        <p style={{
          color: "var(--text-muted)",
          fontSize: "15px",
          lineHeight: 1.6,
          margin: 0,
          maxWidth: "520px",
        }}>
          Введите анамнез и симптомы — система определит наиболее вероятные диагнозы
          на основе официальных протоколов Министерства здравоохранения РК.
        </p>
      </div>

      {/* Form */}
      <div style={{ marginBottom: "32px" }}>
        <SymptomForm
          onResult={setResult}
          onLoading={setLoading}
          onError={setError}
        />
      </div>

      {/* Loading */}
      {loading && (
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: "12px",
          color: "var(--text-muted)",
          fontSize: "14px",
          fontFamily: "'DM Mono', monospace",
        }}>
          <div style={{
            width: "16px", height: "16px",
            border: "2px solid var(--border)",
            borderTopColor: "var(--accent)",
            borderRadius: "50%",
            animation: "spin 0.8s linear infinite",
          }} />
          <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
          Анализ симптомов...
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{
          padding: "14px 18px",
          background: "#1a0a0a",
          border: "1px solid #4a1a1a",
          borderRadius: "8px",
          color: "#ff6b6b",
          fontSize: "14px",
          fontFamily: "'DM Mono', monospace",
        }}>
          Ошибка: {error}
        </div>
      )}

      {/* Results */}
      {result && !loading && (
        <ResultsList data={result} />
      )}

      {/* Disclaimer */}
      <div style={{
        marginTop: "64px",
        paddingTop: "24px",
        borderTop: "1px solid var(--border)",
        color: "var(--text-muted)",
        fontSize: "12px",
        lineHeight: 1.6,
      }}>
        ⚠️ Система предназначена исключительно для поддержки клинических решений.
        Не заменяет консультацию врача. Основана на протоколах МЗ РК.
      </div>
    </main>
  );
}
