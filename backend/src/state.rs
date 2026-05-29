use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use tokio::sync::RwLock;
use sqlx::PgPool;
use crate::config::Config;

#[derive(Clone)]
pub struct AppState {
    pub db: PgPool,
    pub config: Arc<Config>,
    pub openai_key: Arc<RwLock<Option<String>>>,
    pub user_openai_keys: Arc<RwLock<HashMap<i64, String>>>,
    pub fts_ready: Arc<AtomicBool>,
}

impl AsRef<AppState> for AppState {
    fn as_ref(&self) -> &AppState {
        self
    }
}

impl AppState {
    pub fn new(db: PgPool, config: Config, fts_ready: Arc<AtomicBool>) -> Self {
        let openai_key = config.openai_api_key.clone();
        Self {
            db,
            config: Arc::new(config),
            openai_key: Arc::new(RwLock::new(openai_key)),
            user_openai_keys: Arc::new(RwLock::new(HashMap::new())),
            fts_ready,
        }
    }

    pub async fn get_app_mode(&self) -> Option<String> {
        if self.config.app_mode.is_some() {
            return self.config.app_mode.clone();
        }
        sqlx::query_scalar("SELECT value FROM settings WHERE key='app_mode'")
            .fetch_optional(&self.db)
            .await
            .ok()
            .flatten()
    }

    pub async fn get_openai_key(&self, user_id: Option<i64>) -> Option<String> {
        if let Some(uid) = user_id {
            let keys = self.user_openai_keys.read().await;
            if let Some(k) = keys.get(&uid) {
                return Some(k.clone());
            }
        }
        self.openai_key.read().await.clone()
    }

    pub fn get_embedding_model(&self) -> String {
        "text-embedding-3-small".to_string()
    }

    /// Query ChromaDB for the total number of vectors in the active collection.
    /// Returns None if ChromaDB is unreachable or the collection doesn't exist.
    pub async fn get_vector_count(&self) -> Option<i64> {
        let base = format!(
            "http://{}:{}/api/v2/tenants/default_tenant/databases/default_database",
            self.config.chroma_host, self.config.chroma_port
        );
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .ok()?;

        // Step 1: resolve collection name → UUID
        let info: serde_json::Value = client
            .get(format!("{}/collections/{}", base, self.config.chroma_collection))
            .send()
            .await
            .ok()?
            .json()
            .await
            .ok()?;
        let collection_id = info["id"].as_str()?.to_string();

        // Step 2: fetch count for that collection UUID
        let count_val: serde_json::Value = client
            .get(format!("{}/collections/{}/count", base, collection_id))
            .send()
            .await
            .ok()?
            .json()
            .await
            .ok()?;

        count_val.as_i64().or_else(|| count_val.as_f64().map(|f| f as i64))
    }
}
