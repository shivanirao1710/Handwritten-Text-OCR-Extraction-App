export interface Ticket {
  id: number;
  image_url: string;
  created_at: string;
  raw_text_content: string;
  ticket_number: string | null;
  ticket_date: string | null;
  haul_vendor: string | null;
  truck_number: string | null;
  material: string | null;
  job_number: string | null;
  phase_code: string | null;
  zone: string | null;
  hours: number | null;
}