export interface Message {
  id: number
  msg_uuid?: string
  username: string
  date: string
  content: string
  attachments?: string
  reactions?: string
  is_suno_team?: string | boolean
  upload_id?: string
  row_index?: number
  similarity?: number
}

export interface Upload {
  id: string
  filename: string
  row_count: number
  upload_time: string
  embedded_models?: Record<string, boolean>
}

export interface User {
  id: number
  username: string
  is_admin: boolean
  created_at?: string
}

export interface Bookmark {
  id: number
  msg_id: number
  note?: string
  created_at: string
  content?: string
  username?: string
  date?: string
  is_suno_team?: string | boolean
  codes: Code[]
}

export interface Code {
  id: number
  name: string
  color: string
  description?: string
  category_id?: number | null
}

export interface CodeCategory {
  id: number
  name: string
  color?: string
  parent_id?: number | null
  description?: string
}

export interface BookmarkCode {
  bookmark_id: number
  code_id: number
  code_name: string
  color: string
  highlighted_text?: string
  type?: 'excerpt' | 'highlight'
}

export interface Stats {
  total_messages: number
  total_uploads: number
  embedded_messages: number
  api_key_set: boolean
  current_model?: string
}

export interface EmbeddingModel {
  id: string
  label: string
  description: string
  dims: number
  embedded_count: number
  active: boolean
  local: boolean
  available?: boolean
  vector_count?: number
}

export interface SunoTeamMember {
  username: string
  msg_count: number
}

export interface UserInRange {
  username: string
  total_messages: number
  is_suno_team?: boolean
  first_date?: string
  last_date?: string
  avg_words?: number
  weeks_active?: number
  total_weeks?: number
  pct_weeks?: number
}

export interface SearchFilters {
  q?: string
  date_from?: string
  date_to?: string
  upload_ids?: string[]
  is_suno_team?: boolean
  min_words?: number
  limit?: number
  offset?: number
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: Message[]
  isStreaming?: boolean
}

export interface TrendDataPoint {
  period: string
  count: number
}

export interface ContextMessage {
  id: number
  username: string
  date: string
  content: string
  is_target?: boolean
  is_suno_team?: string | boolean
}
