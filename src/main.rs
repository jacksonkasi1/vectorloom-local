use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Multipart, Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post},
};
use image::GenericImageView;
use serde::{Deserialize, Serialize};
use std::{path::PathBuf, sync::Arc, time::Instant};
use tower_http::{cors::CorsLayer, services::ServeDir, trace::TraceLayer};
use tracing::info;
use vectorloom_local::{
    RuntimeStatus, VectorizedImage,
    models::{ModelCatalog, ModelKind, ModelManager},
    runtime_status,
    starvector::StarVectorRuntime,
    vectorize,
};

#[derive(Clone)]
struct AppState {
    models: Arc<ModelManager>,
    starvector: Arc<StarVectorRuntime>,
}

#[derive(Serialize)]
struct HealthResponse {
    status: RuntimeStatus,
}

#[derive(Serialize)]
struct ApiError {
    error: String,
}

#[derive(Deserialize)]
struct SelectModel {
    model: ModelKind,
}

#[derive(Serialize)]
struct Accepted {
    accepted: bool,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let model_manager = Arc::new(ModelManager::new(model_root()));
    if std::env::args().any(|argument| argument == "--bootstrap-models") {
        info!("ensuring both StarVector checkpoints are installed");
        model_manager
            .bootstrap_all()
            .await
            .expect("bootstrap StarVector checkpoints");
        return;
    }
    if matches!(std::env::var("VECTOR_AUTO_DOWNLOAD").as_deref(), Ok("all")) {
        let bootstrap_models = Arc::clone(&model_manager);
        tokio::spawn(async move {
            if let Err(error) = bootstrap_models.bootstrap_all().await {
                tracing::error!(%error, "automatic model bootstrap failed");
            }
        });
    }
    let state = AppState {
        models: model_manager,
        starvector: Arc::new(StarVectorRuntime::new()),
    };
    let app = Router::new()
        .route("/api/health", get(health))
        .route("/api/vectorize", post(vectorize_upload))
        .route("/api/models", get(models))
        .fallback_service(ServeDir::new("web").append_index_html_on_directories(true))
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::very_permissive());
    let app = if std::env::var_os("VECTOR_ENABLE_MODEL_ADMIN").is_some() {
        app.route("/api/models/select", post(select_model))
            .route("/api/models/{model}/download", post(download_model))
            .route("/api/models/{model}", delete(delete_model))
    } else {
        app
    }
    .with_state(state);
    let port = std::env::var("VECTOR_PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(3000);
    let bind_address = std::env::var("VECTOR_BIND").unwrap_or_else(|_| "127.0.0.1".to_owned());
    let listener = tokio::net::TcpListener::bind((bind_address.as_str(), port))
        .await
        .expect("bind server");
    info!("VectorLoom listening at http://{bind_address}:{port}");
    axum::serve(listener, app).await.expect("serve local app");
}

fn model_root() -> PathBuf {
    std::env::var_os("VECTOR_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models"))
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: runtime_status(),
    })
}

async fn models(State(state): State<AppState>) -> Json<ModelCatalog> {
    Json(state.models.catalog().await)
}

async fn select_model(
    State(state): State<AppState>,
    Json(selection): Json<SelectModel>,
) -> Json<ModelCatalog> {
    state.models.select(selection.model);
    Json(state.models.catalog().await)
}

async fn download_model(
    State(state): State<AppState>,
    Path(model): Path<String>,
) -> Result<(StatusCode, Json<Accepted>), (StatusCode, Json<ApiError>)> {
    let kind = parse_model(&model)?;
    state.models.select(kind);
    state
        .models
        .start_download(kind)
        .map_err(|error| conflict(&error.to_string()))?;
    Ok((StatusCode::ACCEPTED, Json(Accepted { accepted: true })))
}

async fn delete_model(
    State(state): State<AppState>,
    Path(model): Path<String>,
) -> Result<Json<ModelCatalog>, (StatusCode, Json<ApiError>)> {
    let kind = parse_model(&model)?;
    state.starvector.unload(kind);
    state
        .models
        .delete(kind)
        .await
        .map_err(|error| conflict(&error.to_string()))?;
    Ok(Json(state.models.catalog().await))
}

async fn vectorize_upload(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let processing_started = Instant::now();
    let mut uploaded_image = None;
    let mut requested_model = None;
    while let Some(field) = multipart.next_field().await.map_err(api_error)? {
        match field.name() {
            Some("image") => {
                uploaded_image = Some((
                    field.content_type().unwrap_or("").to_owned(),
                    field.bytes().await.map_err(api_error)?,
                ))
            }
            Some("model") => requested_model = Some(field.text().await.map_err(api_error)?),
            _ => {}
        }
    }
    let (content_type, bytes) =
        uploaded_image.ok_or_else(|| invalid("Select a PNG, JPEG, or WebP image."))?;
    if !matches!(
        content_type.as_str(),
        "image/png" | "image/jpeg" | "image/webp"
    ) {
        return Err(invalid("Only PNG, JPEG, and WebP files are accepted."));
    }
    let kind = requested_model
        .as_deref()
        .map(parse_model)
        .transpose()?
        .unwrap_or_else(|| state.models.selected());
    let installed = state.models.is_installed(kind).await;
    let model_dir = state.models.model_dir(kind);
    let runtime = Arc::clone(&state.starvector);
    let mut result = tokio::task::spawn_blocking(move || {
        if installed {
            match runtime.generate(kind, &model_dir, &bytes) {
                Ok(generated) => {
                    let image = image::load_from_memory(&bytes)?;
                    let (width, height) = image.dimensions();
                    return Ok::<VectorizedImage, anyhow::Error>(VectorizedImage {
                        svg: generated.svg,
                        width,
                        height,
                        elapsed_ms: generated.elapsed_ms,
                        engine: generated.engine,
                        status: runtime_status(),
                        warning: None,
                    });
                }
                Err(error) => {
                    tracing::warn!(
                        model = %kind.label(),
                        error = %format!("{error:#}"),
                        "StarVector inference failed; using VTracer fallback"
                    );
                    let mut fallback = vectorize(&bytes)?;
                    fallback.warning = Some(format!(
                        "{} inference failed, so the verified local tracer was used: {error:#}",
                        kind.label()
                    ));
                    return Ok(fallback);
                }
            }
        }
        let mut fallback = vectorize(&bytes)?;
        fallback.warning = Some(format!(
            "{} is not downloaded; using the verified local tracer.",
            kind.label()
        ));
        Ok(fallback)
    })
    .await
    .map_err(api_error)?
    .map_err(api_error)?;
    result.elapsed_ms = processing_started.elapsed().as_millis();
    Ok(Json(result))
}

fn parse_model(value: &str) -> Result<ModelKind, (StatusCode, Json<ApiError>)> {
    match value {
        "1b" => Ok(ModelKind::OneB),
        "8b" => Ok(ModelKind::EightB),
        _ => Err(invalid("Unknown model. Choose 1b or 8b.")),
    }
}

fn invalid(message: &str) -> (StatusCode, Json<ApiError>) {
    (
        StatusCode::BAD_REQUEST,
        Json(ApiError {
            error: message.to_owned(),
        }),
    )
}

fn conflict(message: &str) -> (StatusCode, Json<ApiError>) {
    (
        StatusCode::CONFLICT,
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
