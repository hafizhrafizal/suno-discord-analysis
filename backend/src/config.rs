use std::env;

#[derive(Clone, Debug)]
pub struct Config {
    pub db_path: String,
    pub chroma_host: String,
    pub chroma_port: u16,
    pub chroma_collection: String,
    pub openai_api_key: Option<String>,
    pub app_mode: Option<String>,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            db_path: env::var("DB_PATH").unwrap_or_else(|_| "../discord_data.db".to_string()),
            chroma_host: env::var("CHROMA_HOST").unwrap_or_else(|_| "localhost".to_string()),
            chroma_port: env::var("CHROMA_PORT").ok().and_then(|v| v.parse().ok()).unwrap_or(8001),
            chroma_collection: env::var("CHROMA_COLLECTION").unwrap_or_else(|_| "discord_openai".to_string()),
            openai_api_key: env::var("OPENAI_API_KEY").ok(),
            app_mode: env::var("APP_MODE").ok(),
        }
    }
}
