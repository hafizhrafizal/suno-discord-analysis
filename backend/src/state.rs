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
}
