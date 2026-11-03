interface TextEditorProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
  selectedModel: "llama" | "t5" | "coedit";
  onModelChange: (model: "llama" | "t5" | "coedit") => void;
}

const MAX_CHARACTERS = 5000;

function TextEditor({ value, onChange, onSubmit, isLoading, selectedModel, onModelChange }: TextEditorProps) {
  const characterCount = value.length;
  const isOverLimit = characterCount > MAX_CHARACTERS;
  const isEmpty = value.trim().length === 0;

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && !isLoading && !isEmpty && !isOverLimit) {
      e.preventDefault();
      onSubmit();
    }
  };

  return (
    <div className="card">
      <label htmlFor="text-input" className="block text-sm font-medium text-gray-700">
        Enter your text
      </label>

      <textarea
        id="text-input"
        rows={8}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Paste or type the text you want to check for grammar errors..."
        disabled={isLoading}
        className={`mt-2 block w-full resize-y rounded-lg border bg-white px-4 py-3
          text-sm leading-relaxed text-gray-900 placeholder-gray-400
          transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500
          disabled:cursor-not-allowed disabled:bg-gray-50 disabled:text-gray-500
          ${isOverLimit ? "border-red-300 focus:ring-red-500" : "border-gray-300"}`}
      />

      <div className="mt-3 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <span
            className={`text-xs ${
              isOverLimit ? "font-medium text-red-600" : "text-gray-500"
            }`}
          >
            {characterCount.toLocaleString()} / {MAX_CHARACTERS.toLocaleString()} characters
          </span>
          <span className="text-xs text-gray-400">
            Ctrl+Enter to submit
          </span>
        </div>

        <div className="flex items-center gap-3">
          <select
            value={selectedModel}
            onChange={(e) => onModelChange(e.target.value as "llama" | "t5" | "coedit")}
            disabled={isLoading}
            className="rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-700
              transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500
              disabled:cursor-not-allowed disabled:bg-gray-50 disabled:text-gray-500"
          >
            <option value="llama">Llama 3.2-3B (Fine-tuned)</option>
            <option value="coedit">CoEdIT (Grammarly)</option>
            <option value="t5">T5 (Fine-tuned)</option>
          </select>

          <button
            type="button"
            onClick={onSubmit}
            disabled={isLoading || isEmpty || isOverLimit}
            className="btn-primary"
          >
          {isLoading ? (
            <>
              <svg
                className="-ml-1 mr-2 h-4 w-4 animate-spin text-white"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                />
              </svg>
              Checking...
            </>
          ) : (
            "Check Grammar"
          )}
        </button>
        </div>
      </div>
    </div>
  );
}

export default TextEditor;
