import { useState, useCallback } from "react";
import Header from "@/components/Header";
import TextEditor from "@/components/TextEditor";
import CorrectionPanel from "@/components/CorrectionPanel";
import { correctText } from "@/services/api";
import type { CorrectionResponse } from "@/types";

function App() {
  const [inputText, setInputText] = useState("");
  const [selectedModel, setSelectedModel] = useState<"llama" | "t5" | "coedit">("llama");
  const [result, setResult] = useState<CorrectionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = useCallback(async () => {
    if (!inputText.trim()) return;

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await correctText(inputText, undefined, selectedModel);
      setResult(response);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "An unexpected error occurred.";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, [inputText, selectedModel]);

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />

      <main className="mx-auto max-w-5xl px-4 py-8 sm:px-6 lg:px-8">
        <div className="space-y-8">
          <TextEditor
            value={inputText}
            onChange={setInputText}
            onSubmit={handleSubmit}
            isLoading={isLoading}
            selectedModel={selectedModel}
            onModelChange={setSelectedModel}
          />

          {error && (
            <div
              role="alert"
              className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700"
            >
              <p className="font-medium">Error</p>
              <p className="mt-1">{error}</p>
            </div>
          )}

          <CorrectionPanel result={result} />
        </div>
      </main>
    </div>
  );
}

export default App;
