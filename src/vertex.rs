use crate::circuit_breaker::CircuitBreaker;
use crate::connectors::{
    to_gemini_body, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice, Connector,
    ConnectorError, ConnectorResult, Provider, Usage,
};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD, Engine};
use dashmap::DashMap;
use jsonwebtoken::{encode, Algorithm, EncodingKey, Header};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::Mutex;

const CLOUD_PLATFORM_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";
const REFRESH_SKEW_SECS: u64 = 300;
const MAX_CACHE_ENTRIES: usize = 128;

#[derive(Debug, Deserialize)]
struct ServiceAccount {
    client_email: String,
    private_key: String,
    token_uri: String,
}

#[derive(Debug, Serialize)]
struct JwtClaims<'a> {
    iss: &'a str,
    scope: &'a str,
    aud: &'a str,
    iat: u64,
    exp: u64,
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    expires_in: u64,
}

#[derive(Clone)]
struct CachedToken {
    access_token: String,
    expires_at: u64,
}

enum VertexCredential {
    ServiceAccount(ServiceAccount),
    ApiKey(String),
}

pub struct VertexConnection {
    pub project: String,
    pub location: String,
    credential: VertexCredential,
    cache_key: String,
}

pub enum VertexAuth {
    Bearer(String),
    ApiKey(String),
}

pub struct VertexConnector {
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
    tokens: DashMap<String, Arc<Mutex<Option<CachedToken>>>>,
}

impl VertexConnector {
    pub fn new(circuit_breaker: Arc<CircuitBreaker>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("Vertex HTTP client"),
            circuit_breaker,
            tokens: DashMap::new(),
        }
    }

    pub fn parse_connection(raw: &str) -> Result<VertexConnection, ConnectorError> {
        let mut project = None;
        let mut location = None;
        let mut credentials_base64 = None;
        let mut api_key = None;
        for part in raw.split(';') {
            if let Some((key, value)) = part.trim().split_once('=') {
                match key.trim().to_ascii_lowercase().as_str() {
                    "project" => project = Some(value.trim().to_owned()),
                    "location" | "region" => location = Some(value.trim().to_owned()),
                    "credentials_base64" => credentials_base64 = Some(value.trim().to_owned()),
                    "api_key" => api_key = Some(value.trim().to_owned()),
                    _ => {}
                }
            }
        }
        let project = project.ok_or_else(|| {
            ConnectorError::BadResponse("Vertex connection missing 'project='".into())
        })?;
        let location = location.ok_or_else(|| {
            ConnectorError::BadResponse("Vertex connection missing 'location='".into())
        })?;
        if credentials_base64.is_some() == api_key.is_some() {
            return Err(ConnectorError::BadResponse("Vertex connection requires exactly one of 'credentials_base64=' or testing-only 'api_key='".into()));
        }
        let (credential, secret_material) = if let Some(encoded) = credentials_base64 {
            let bytes = STANDARD.decode(&encoded).map_err(|_| {
                ConnectorError::BadResponse("Vertex credentials_base64 is not valid base64".into())
            })?;
            let account: ServiceAccount = serde_json::from_slice(&bytes).map_err(|_| {
                ConnectorError::BadResponse(
                    "Vertex credentials_base64 is not valid service-account JSON".into(),
                )
            })?;
            if account.token_uri != "https://oauth2.googleapis.com/token" {
                return Err(ConnectorError::BadResponse(
                    "Vertex service-account token_uri must be https://oauth2.googleapis.com/token"
                        .into(),
                ));
            }
            (VertexCredential::ServiceAccount(account), encoded)
        } else {
            let key = api_key.unwrap();
            (VertexCredential::ApiKey(key.clone()), key)
        };
        let mut hasher = Sha256::new();
        hasher.update(project.as_bytes());
        hasher.update(location.as_bytes());
        hasher.update(secret_material.as_bytes());
        let cache_key = format!("{:x}", hasher.finalize());
        Ok(VertexConnection {
            project,
            location,
            credential,
            cache_key,
        })
    }

    pub fn url(
        connection: &VertexConnection,
        model: &str,
        streaming: bool,
    ) -> Result<String, ConnectorError> {
        let valid_project = connection
            .project
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
        let valid_location = connection
            .location
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-');
        let valid_model = model
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '@'));
        if !valid_project || !valid_location || !valid_model {
            return Err(ConnectorError::BadResponse(
                "Vertex project, location, or model contains unsupported URL characters".into(),
            ));
        }
        let method = if streaming {
            "streamGenerateContent?alt=sse"
        } else {
            "generateContent"
        };
        Ok(format!("https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/{}:{}", connection.location, connection.project, connection.location, model, method))
    }

    pub async fn auth(
        &self,
        connection: &VertexConnection,
        force_refresh: bool,
    ) -> Result<VertexAuth, ConnectorError> {
        match &connection.credential {
            VertexCredential::ApiKey(key) => Ok(VertexAuth::ApiKey(key.clone())),
            VertexCredential::ServiceAccount(account) => {
                if self.tokens.len() >= MAX_CACHE_ENTRIES
                    && !self.tokens.contains_key(&connection.cache_key)
                {
                    if let Some(key) = self.tokens.iter().next().map(|entry| entry.key().clone()) {
                        self.tokens.remove(&key);
                    }
                }
                let slot = self
                    .tokens
                    .entry(connection.cache_key.clone())
                    .or_insert_with(|| Arc::new(Mutex::new(None)))
                    .clone();
                let mut guard = slot.lock().await;
                let now = unix_now();
                if !force_refresh {
                    if let Some(token) = guard.as_ref() {
                        if token.expires_at > now + REFRESH_SKEW_SECS {
                            return Ok(VertexAuth::Bearer(token.access_token.clone()));
                        }
                    }
                }
                let claims = JwtClaims {
                    iss: &account.client_email,
                    scope: CLOUD_PLATFORM_SCOPE,
                    aud: &account.token_uri,
                    iat: now,
                    exp: now + 3600,
                };
                let assertion = encode(
                    &Header::new(Algorithm::RS256),
                    &claims,
                    &EncodingKey::from_rsa_pem(account.private_key.as_bytes()).map_err(|_| {
                        ConnectorError::BadResponse(
                            "Vertex service-account private_key is invalid".into(),
                        )
                    })?,
                )
                .map_err(|_| {
                    ConnectorError::BadResponse("Could not sign Vertex OAuth assertion".into())
                })?;
                let response = self
                    .client
                    .post(&account.token_uri)
                    .form(&[
                        ("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
                        ("assertion", assertion.as_str()),
                    ])
                    .send()
                    .await?;
                if !response.status().is_success() {
                    return Err(ConnectorError::Unauthorized);
                }
                let token: TokenResponse = response.json().await.map_err(|e| {
                    ConnectorError::BadResponse(format!("Invalid Google OAuth response: {e}"))
                })?;
                *guard = Some(CachedToken {
                    access_token: token.access_token.clone(),
                    expires_at: now + token.expires_in,
                });
                Ok(VertexAuth::Bearer(token.access_token))
            }
        }
    }

    fn request(&self, url: &str, auth: &VertexAuth) -> reqwest::RequestBuilder {
        let request = self
            .client
            .post(url)
            .header("content-type", "application/json");
        match auth {
            VertexAuth::Bearer(token) => request.bearer_auth(token),
            VertexAuth::ApiKey(key) => request.header("x-goog-api-key", key),
        }
    }
}

#[derive(Deserialize)]
struct VertexResponse {
    #[serde(default)]
    candidates: Vec<VertexCandidate>,
    #[serde(rename = "usageMetadata")]
    usage: Option<VertexUsage>,
}
#[derive(Deserialize)]
struct VertexCandidate {
    content: VertexContent,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}
#[derive(Deserialize)]
struct VertexContent {
    #[serde(default)]
    parts: Vec<VertexPart>,
}
#[derive(Deserialize)]
struct VertexPart {
    text: Option<String>,
}
#[derive(Deserialize)]
struct VertexUsage {
    #[serde(rename = "promptTokenCount", default)]
    input: u32,
    #[serde(rename = "candidatesTokenCount", default)]
    output: u32,
}

#[async_trait]
impl Connector for VertexConnector {
    async fn complete(
        &self,
        req: &ChatCompletionRequest,
        raw: &str,
    ) -> Result<ConnectorResult, ConnectorError> {
        let started = Instant::now();
        if self.circuit_breaker.is_open(Provider::VertexAI) {
            return Err(ConnectorError::CircuitOpen);
        }
        let connection = Self::parse_connection(raw)?;
        let url = Self::url(&connection, &req.model, false)?;
        let body = to_gemini_body(req);
        let mut auth = self.auth(&connection, false).await?;
        let mut response = self.request(&url, &auth).json(&body).send().await?;
        if response.status().as_u16() == 401 && matches!(auth, VertexAuth::Bearer(_)) {
            auth = self.auth(&connection, true).await?;
            response = self.request(&url, &auth).json(&body).send().await?;
        }
        let status = response.status().as_u16();
        if status == 401 || status == 403 {
            return Err(ConnectorError::Unauthorized);
        }
        if status == 429 {
            return Err(ConnectorError::RateLimited);
        }
        if status >= 500 {
            self.circuit_breaker.record_failure(Provider::VertexAI);
            return Err(ConnectorError::ServerError { status });
        }
        let text = response.text().await?;
        if !(200..300).contains(&status) {
            return Err(ConnectorError::BadResponse(format!(
                "HTTP {status}: {text}"
            )));
        }
        let parsed: VertexResponse = serde_json::from_str(&text)
            .map_err(|e| ConnectorError::BadResponse(format!("Unexpected Vertex response: {e}")))?;
        let content = parsed
            .candidates
            .first()
            .and_then(|c| c.content.parts.first())
            .and_then(|p| p.text.clone())
            .unwrap_or_default();
        let finish = parsed
            .candidates
            .first()
            .and_then(|c| c.finish_reason.clone())
            .unwrap_or_else(|| "stop".into());
        let (input, output) = parsed
            .usage
            .map(|u| (u.input, u.output))
            .unwrap_or_default();
        self.circuit_breaker.record_success(Provider::VertexAI);
        Ok(ConnectorResult {
            provider: Provider::VertexAI,
            model_id: req.model.clone(),
            input_tokens: input,
            output_tokens: output,
            latency_ms: started.elapsed().as_millis() as u64,
            response: ChatCompletionResponse {
                id: format!("vertex-{}", unix_now()),
                object: "chat.completion".into(),
                created: unix_now(),
                model: req.model.clone(),
                choices: vec![Choice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".into(),
                        content: crate::vision::MessageContent::Text(content),
                    },
                    finish_reason: finish,
                }],
                usage: Usage {
                    prompt_tokens: input,
                    completion_tokens: output,
                    total_tokens: input + output,
                },
            },
        })
    }
    fn provider(&self) -> Provider {
        Provider::VertexAI
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_testing_api_key_connection() {
        let parsed =
            VertexConnector::parse_connection("project=p;location=us-central1;api_key=test")
                .unwrap();
        assert_eq!(parsed.project, "p");
        assert_eq!(parsed.location, "us-central1");
    }
    #[test]
    fn rejects_ambiguous_auth() {
        assert!(VertexConnector::parse_connection(
            "project=p;location=l;api_key=k;credentials_base64=e30="
        )
        .is_err());
    }

    #[test]
    fn vertex_url_is_project_and_location_scoped() {
        let parsed = VertexConnector::parse_connection(
            "project=my-project;location=us-central1;api_key=test",
        )
        .unwrap();
        assert_eq!(VertexConnector::url(&parsed, "gemini-2.5-pro", false).unwrap(), "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-2.5-pro:generateContent");
    }
}
