mod config;
mod database;
mod error;
pub mod hdbscan;
mod middleware;
mod models;
pub mod prompts;
mod routes;
mod state;

use axum::{
    http::{Method, StatusCode},
    routing::get_service,
    Router,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tower_http::{
    cors::{Any, CorsLayer},
    services::ServeDir,
    trace::TraceLayer,
};
use tower_sessions::{Expiry, SessionManagerLayer};
use tower_sessions_sqlx_store::SqliteStore;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use config::Config;
use database::{
    backfill_precomputed_columns, create_pool, init_db, needs_fts_rebuild, needs_word_count_backfill,
    rebuild_fts, seed_demo_user,
};
use state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load .env from repo root (one level up from backend/)
    dotenvy::from_path("../.env").ok();

    // Init tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "retrieval_backend=debug,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cfg = Config::from_env();
    tracing::info!("Starting retrieval-backend — DB: {}", cfg.db_path);

    // Create DB pool
    let pool = create_pool(&cfg.db_path).await?;
    init_db(&pool).await?;
    tracing::info!("Database initialized");

    // FTS5 readiness flag — set true immediately if index is already in sync,
    // or after the background rebuild completes.
    let fts_ready = Arc::new(AtomicBool::new(false));
    if needs_fts_rebuild(&pool).await {
        let pool_fts = pool.clone();
        let flag = fts_ready.clone();
        tokio::spawn(async move { rebuild_fts(&pool_fts, flag).await });
    } else {
        fts_ready.store(true, Ordering::Relaxed);
        tracing::info!("FTS5 index is current — fast keyword search active from startup");
    }

    // Backfill precomputed columns (word_count, week) for existing rows if needed
    if needs_word_count_backfill(&pool).await {
        let pool_bf = pool.clone();
        tokio::spawn(async move { backfill_precomputed_columns(&pool_bf).await });
    }

    // Seed default user for demo/single mode (no login required)
    let mode = cfg.app_mode.as_deref().unwrap_or("");
    if mode == "demo" || mode == "single" {
        seed_demo_user(&pool).await?;
        tracing::info!("Mode={}: auto-login enabled", mode);
    }

    // Session store (uses same SQLite DB)
    let session_store = SqliteStore::new(pool.clone());
    session_store.migrate().await?;

    let session_layer = SessionManagerLayer::new(session_store)
        .with_secure(false)
        .with_expiry(Expiry::OnInactivity(Duration::from_secs(30 * 24 * 3600).try_into().unwrap()));

    // App state
    let state = AppState::new(pool, cfg, fts_ready);

    // CORS
    let cors = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE, Method::OPTIONS])
        .allow_headers(Any)
        .allow_origin(Any);

    // Static file service
    let static_service = get_service(ServeDir::new("../static")).handle_error(|_| async move {
        (StatusCode::INTERNAL_SERVER_ERROR, "static file error")
    });

    // Frontend dist (served if built)
    let frontend_service = get_service(ServeDir::new("../frontend/dist")).handle_error(|_| async move {
        (StatusCode::NOT_FOUND, "frontend not built")
    });

    let app = Router::new()
        .nest("/api", routes::all_routes(state))
        .nest_service("/static", static_service)
        .fallback_service(frontend_service)
        .layer(cors)
        .layer(session_layer)
        .layer(TraceLayer::new_for_http());

    let addr = "0.0.0.0:8000";
    tracing::info!("Listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
