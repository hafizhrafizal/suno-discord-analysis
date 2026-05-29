use std::env;

#[derive(Clone, Debug)]
pub struct Config {
    pub database_url: String,
    pub vector_db: String,
    // ChromaDB
    pub chroma_host: String,
    pub chroma_port: u16,
    pub chroma_collection: String,
    // Qdrant
    pub qdrant_url: String,
    pub qdrant_collection: String,
    pub qdrant_api_key: Option<String>,
    pub openai_api_key: Option<String>,
    pub app_mode: Option<String>,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            database_url: env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgres://postgres:password@localhost/discord_db".to_string()),
            vector_db: env::var("VECTOR_DB").unwrap_or_else(|_| "chroma_http".to_string()),
            chroma_host: env::var("CHROMA_HOST").unwrap_or_else(|_| "localhost".to_string()),
            chroma_port: env::var("CHROMA_PORT").ok().and_then(|v| v.parse().ok()).unwrap_or(8001),
            chroma_collection: env::var("CHROMA_COLLECTION")
                .unwrap_or_else(|_| "discord_openai".to_string()),
            qdrant_url: env::var("QDRANT_URL")
                .unwrap_or_else(|_| "http://localhost:6333".to_string()),
            qdrant_collection: env::var("QDRANT_COLLECTION")
                .unwrap_or_else(|_| "discord_openai".to_string()),
            qdrant_api_key: env::var("QDRANT_API_KEY").ok(),
            openai_api_key: env::var("OPENAI_API_KEY").ok(),
            app_mode: env::var("APP_MODE").ok(),
        }
    }

    pub fn vector_db_label(&self) -> &'static str {
        match self.vector_db.to_lowercase().as_str() {
            "qdrant" => "Qdrant",
            _ => "ChromaDB",
        }
    }
}
