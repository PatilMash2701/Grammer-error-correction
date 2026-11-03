import type { CorrectionResponse } from "@/types";

interface CorrectionPanelProps {
  result: CorrectionResponse | null;
}

/** Map error type strings to display-friendly labels and color classes. */
function getErrorBadge(errorType: string): { label: string; className: string } {
  const normalized = errorType.toLowerCase().replace(/[_\s]+/g, " ");

  const mapping: Record<string, { label: string; className: string }> = {
    spelling: {
      label: "Spelling",
      className: "bg-red-100 text-red-700",
    },
    grammar: {
      label: "Grammar",
      className: "bg-orange-100 text-orange-700",
    },
    punctuation: {
      label: "Punctuation",
      className: "bg-yellow-100 text-yellow-700",
    },
    style: {
      label: "Style",
      className: "bg-blue-100 text-blue-700",
    },
    "word choice": {
      label: "Word Choice",
      className: "bg-purple-100 text-purple-700",
    },
    capitalization: {
      label: "Capitalization",
      className: "bg-teal-100 text-teal-700",
    },
  };

  const match = Object.entries(mapping).find(([key]) =>
    normalized.includes(key)
  );

  return match
    ? match[1]
    : { label: errorType, className: "bg-gray-100 text-gray-700" };
}

/** Render confidence as a percentage with a colored indicator. */
function ConfidenceIndicator({ score }: { score: number }) {
  const percentage = Math.round(score * 100);

  let colorClass: string;
  if (percentage >= 90) {
    colorClass = "text-green-600 bg-green-50 border-green-200";
  } else if (percentage >= 70) {
    colorClass = "text-yellow-600 bg-yellow-50 border-yellow-200";
  } else {
    colorClass = "text-red-600 bg-red-50 border-red-200";
  }

  return (
    <span
      className={`inline-flex items-center rounded-full border px-3 py-1 text-sm font-medium ${colorClass}`}
    >
      Confidence: {percentage}%
    </span>
  );
}

function CorrectionPanel({ result }: CorrectionPanelProps) {
  if (!result) return null;

  const hasCorrections = result.corrections.length > 0;

  return (
    <div className="space-y-6">
      {/* Summary bar */}
      <div className="card flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <ConfidenceIndicator score={result.confidence_score} />
          <span className="text-sm text-gray-500">
            {result.corrections.length}{" "}
            {result.corrections.length === 1 ? "correction" : "corrections"} found
          </span>
        </div>
        <span className="text-xs text-gray-400">
          Processed in {result.processing_time_ms.toFixed(0)}ms
        </span>
      </div>

      {/* Corrected text */}
      <div className="card">
        <h2 className="text-sm font-semibold text-gray-700">Corrected Text</h2>
        {hasCorrections ? (
          <p className="mt-3 whitespace-pre-wrap rounded-lg bg-green-50 p-4 text-sm leading-relaxed text-gray-800">
            {result.corrected_text}
          </p>
        ) : (
          <p className="mt-3 rounded-lg bg-gray-50 p-4 text-sm leading-relaxed text-gray-600">
            No corrections needed -- your text looks good!
          </p>
        )}
      </div>

      {/* Individual corrections */}
      {hasCorrections && (
        <div className="card">
          <h2 className="text-sm font-semibold text-gray-700">
            Corrections Detail
          </h2>

          <ul className="mt-4 divide-y divide-gray-100">
            {result.corrections.map((correction, index) => {
              const badge = getErrorBadge(correction.error_type);

              return (
                <li key={index} className="py-3 first:pt-0 last:pb-0">
                  <div className="flex flex-wrap items-start gap-3">
                    <span
                      className={`inline-flex shrink-0 items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${badge.className}`}
                    >
                      {badge.label}
                    </span>

                    <div className="min-w-0 flex-1 text-sm">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="rounded bg-red-50 px-2 py-0.5 font-mono text-red-700 line-through">
                          {correction.original}
                        </span>
                        <svg
                          className="h-4 w-4 shrink-0 text-gray-400"
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          strokeWidth={2}
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M13 7l5 5m0 0l-5 5m5-5H6"
                          />
                        </svg>
                        <span className="rounded bg-green-50 px-2 py-0.5 font-mono text-green-700">
                          {correction.corrected}
                        </span>
                      </div>
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}

export default CorrectionPanel;
