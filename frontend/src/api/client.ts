// ──────────────────────────────────────────────
// TypeScript interfaces
// ──────────────────────────────────────────────

export interface ExhibitGRow {
  id: number;
  row_number: number;
  cast_number: string;
  cast_name: string;
  character: string;
  status_code: string;
  makeup_time: string;
  report_on_set: string;
  dismiss_on_set: string;
  dismiss_mu_hair: string;
  nd_breakfast_out: string;
  nd_breakfast_in: string;
  first_meal_in: string;
  first_meal_out: string;
  second_meal_in: string;
  second_meal_out: string;
  leave_for_loc: string;
  arrive_on_loc: string;
  leave_loc: string;
  arrive_hotel: string;
  stunt_adjust: string;
  mileage: string;
  mpv_fc: string;
}

export interface CellDisagreement {
  id: number;
  row_number: number;
  column_name: string;
  values: Record<string, string>;
  resolved_value: string | null;
  is_resolved: boolean;
}

export interface ExhibitGSummary {
  id: number;
  original_filename: string;
  production_co: string;
  series_title: string;
  date: string;
  status: string;
  created_at: string;
  row_count: number;
  disagreement_count: number;
}

export interface ExhibitGDetail {
  id: number;
  original_filename: string;
  corrected_image_url: string;
  original_image_url: string;
  skew_angle: number | null;
  production_co: string;
  series_title: string;
  episode_name: string;
  prod_number: string;
  date: string;
  day_x_of_y: string;
  contact: string;
  phone: string;
  status: string;
  created_at: string;
  rows: ExhibitGRow[];
  disagreements: CellDisagreement[];
}

export interface EngineReading {
  engine: string;
  value: string;
  confidence: number;
}

// ──────────────────────────────────────────────
// Column definitions for the table display
// ──────────────────────────────────────────────

export interface ColumnDef {
  key: keyof ExhibitGRow;
  label: string;
  group?: string;
  width?: string;
}

export const COLUMN_DEFS: ColumnDef[] = [
  { key: "cast_number", label: "#", width: "40px" },
  { key: "cast_name", label: "CAST", width: "120px" },
  { key: "character", label: "CHARACTER", width: "120px" },
  { key: "status_code", label: "STATUS", width: "50px" },
  { key: "makeup_time", label: "MU/HAIR WRDRBE", group: "WORK TIME", width: "80px" },
  { key: "report_on_set", label: "REPORT ON SET", group: "WORK TIME", width: "80px" },
  { key: "dismiss_on_set", label: "DISMISS ON SET", group: "WORK TIME", width: "80px" },
  { key: "dismiss_mu_hair", label: "DISMISS MU/HAIR", group: "WORK TIME", width: "80px" },
  { key: "nd_breakfast_out", label: "OUT", group: "ND BRKFST", width: "55px" },
  { key: "nd_breakfast_in", label: "IN", group: "ND BRKFST", width: "55px" },
  { key: "first_meal_in", label: "IN", group: "1ST MEAL", width: "55px" },
  { key: "first_meal_out", label: "OUT", group: "1ST MEAL", width: "55px" },
  { key: "second_meal_in", label: "IN", group: "2ND MEAL", width: "55px" },
  { key: "second_meal_out", label: "OUT", group: "2ND MEAL", width: "55px" },
  { key: "leave_for_loc", label: "LEAVE FOR", group: "TRAVEL TIME", width: "70px" },
  { key: "arrive_on_loc", label: "ARRIVE ON", group: "TRAVEL TIME", width: "70px" },
  { key: "leave_loc", label: "LEAVE", group: "TRAVEL TIME", width: "70px" },
  { key: "arrive_hotel", label: "ARRIVE AT", group: "TRAVEL TIME", width: "70px" },
  { key: "stunt_adjust", label: "STUNT ADJ", width: "70px" },
  { key: "mileage", label: "MILEAGE", width: "60px" },
  { key: "mpv_fc", label: "MPV/FC", width: "60px" },
];

// ──────────────────────────────────────────────
// API functions
// ──────────────────────────────────────────────

const BASE_URL = "/api/transcribe";

export async function uploadExhibitG(
  file: File,
  options: { usePaddle?: boolean; useEasyocr?: boolean; useClaude?: boolean } = {}
): Promise<ExhibitGDetail> {
  const formData = new FormData();
  formData.append("file", file);

  const params = new URLSearchParams();
  if (options.usePaddle !== undefined) params.set("use_paddle", String(options.usePaddle));
  if (options.useEasyocr !== undefined) params.set("use_easyocr", String(options.useEasyocr));
  if (options.useClaude !== undefined) params.set("use_claude", String(options.useClaude));

  const url = `${BASE_URL}/upload${params.toString() ? "?" + params.toString() : ""}`;

  const response = await fetch(url, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Upload failed" }));
    throw new Error(error.detail || "Upload failed");
  }

  return response.json();
}

export async function listTranscriptions(): Promise<ExhibitGSummary[]> {
  const response = await fetch(BASE_URL);
  if (!response.ok) throw new Error("Failed to fetch transcriptions");
  return response.json();
}

export async function getTranscription(id: number): Promise<ExhibitGDetail> {
  const response = await fetch(`${BASE_URL}/${id}`);
  if (!response.ok) throw new Error("Transcription not found");
  return response.json();
}

export async function updateRow(
  exhibitId: number,
  rowId: number,
  updates: Partial<ExhibitGRow>
): Promise<ExhibitGRow> {
  const response = await fetch(`${BASE_URL}/${exhibitId}/rows/${rowId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  });
  if (!response.ok) throw new Error("Failed to update row");
  return response.json();
}

export async function updateHeader(
  exhibitId: number,
  updates: Record<string, string>
): Promise<void> {
  const response = await fetch(`${BASE_URL}/${exhibitId}/header`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  });
  if (!response.ok) throw new Error("Failed to update header");
}

export async function resolveDisagreement(
  exhibitId: number,
  disagreementId: number,
  resolvedValue: string
): Promise<{ status: string; remaining_disagreements: number }> {
  const response = await fetch(`${BASE_URL}/${exhibitId}/resolve/${disagreementId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ resolved_value: resolvedValue }),
  });
  if (!response.ok) throw new Error("Failed to resolve disagreement");
  return response.json();
}

export async function getEngineReadings(
  exhibitId: number,
  rowNumber: number,
  columnName: string
): Promise<EngineReading[]> {
  const response = await fetch(`${BASE_URL}/${exhibitId}/engines/${rowNumber}/${columnName}`);
  if (!response.ok) throw new Error("Failed to fetch engine readings");
  return response.json();
}

export function getExportCsvUrl(exhibitId: number): string {
  return `${BASE_URL}/${exhibitId}/export/csv`;
}
