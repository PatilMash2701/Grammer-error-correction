import axios from "axios";
import type { CorrectionResponse, HealthResponse } from "@/types";

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "/api/v1",
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 300_000, // 5 minutes for long paragraphs
});

/**
 * Submit text for grammar correction.
 *
 * @param text - The text to correct.
 * @param numBeams - Number of beams for the model (optional, higher = more accurate but slower).
 * @param model - Which model to use: "llama", "t5", or "coedit" (default: "llama").
 * @returns The correction response containing original text, corrected text, and diffs.
 */
export async function correctText(
  text: string,
  numBeams?: number,
  model?: "llama" | "t5" | "coedit"
): Promise<CorrectionResponse> {
  const response = await apiClient.post<CorrectionResponse>("/correct", {
    text,
    num_beams: numBeams,
    model: model ?? "llama",
  });
  return response.data;
}

/**
 * Check the health status of the backend API.
 *
 * @returns The health response with model status and uptime information.
 */
export async function healthCheck(): Promise<HealthResponse> {
  const response = await apiClient.get<HealthResponse>("/health");
  return response.data;
}

export default apiClient;
