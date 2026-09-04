use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Multipart},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use serde::Serialize;
use tower_http::{cors::CorsLayer, services::ServeDir, trace::TraceLayer};
use tracing::info;
use vectorloom_local::{RuntimeStatus, runtime_status, vectorize};

#[derive(Serialize)]
struct HealthResponse {
    status: RuntimeStatus,
}

#[derive(Serialize)]
struct ApiError {
    error: String,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let app = Router::new()
        .route("/api/health", get(health))
        .route("/api/vectorize", post(vectorize_upload))
        .fallback_service(ServeDir::new("web").append_index_html_on_directories(true))
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::very_permissive());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .expect("bind local server");
    info!("VectorLoom is local-only at http://127.0.0.1:3000");
    axum::serve(listener, app).await.expect("serve local app");
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: runtime_status(),
    })
}

async fn vectorize_upload(
    mut multipart: Multipart,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let field = multipart
        .next_field()
        .await
        .map_err(api_error)?
        .ok_or_else(|| invalid("Select a PNG, JPEG, or WebP image."))?;
    let content_type = field.content_type().unwrap_or("").to_owned();
    if !matches!(
        content_type.as_str(),
        "image/png" | "image/jpeg" | "image/webp"
    ) {
        return Err(invalid("Only PNG, JPEG, and WebP files are accepted."));
    }
    let bytes = field.bytes().await.map_err(api_error)?;
    let result = tokio::task::spawn_blocking(move || vectorize(&bytes))
        .await
        .map_err(api_error)?
        .map_err(api_error)?;
    Ok(Json(result))
}

fn invalid(message: &str) -> (StatusCode, Json<ApiError>) {
    (
        StatusCode::BAD_REQUEST,
        Json(ApiError {
            error: message.to_owned(),
        }),
    )
}

fn api_error(error: impl std::fmt::Display) -> (StatusCode, Json<ApiError>) {
    tracing::warn!(%error, "vectorization request failed");
    (
        StatusCode::UNPROCESSABLE_ENTITY,
        Json(ApiError {
            error: error.to_string(),
        }),
    )
}
