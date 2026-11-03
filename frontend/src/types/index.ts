/** Individual correction diff showing what was changed and why. */
export interface CorrectionDiff {
  original: string;
  corrected: string;
  error_type: string;
  position: {
    start: number;
    end: number;
  };
}

/** Request payload for single text correction. */
export interface CorrectionRequest {
  text: string;
  model?: "t5" | "coedit";
  num_beams?: number;
}

/** Response payload from single text correction. */
export interface CorrectionResponse {
  original_text: string;
  corrected_text: string;
  corrections: CorrectionDiff[];
  confidence_score: number;
  model_used: string;
  processing_time_ms: number;
}

/** Request payload for batch text correction. */
export interface BatchCorrectionRequest {
  texts: string[];
  num_beams?: number;
}

/** Response payload from batch text correction. */
export interface BatchCorrectionResponse {
  results: CorrectionResponse[];
  total_processing_time_ms: number;
}

/** Response from the health check endpoint. */
export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  version: string;
  uptime_seconds: number;
}
