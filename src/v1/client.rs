use super::completion::{CompletionRequest, CompletionResponse};

#[derive(thiserror::Error, Debug)]
pub enum ApiError {
    /// User exceeds his current Task Quota.
    #[error(
        "You are trying to send too many requests to the API in to short an interval. Slow down a \
        bit, otherwise these error will persist. Sorry for this, but we try to prevent DOS attacks."
    )]
    TooManyRequests,
    /// Model is busy. Most likely due to many other users requesting its services right now.
    #[error(
        "Sorry the request to the Aleph Alpha API has been rejected due to the requested model \
        being very busy at the moment. We found it unlikely that your request would finish in a \
        reasonable timeframe, so it was rejected right away, rather than make you wait. You are \
        welcome to retry your request any time."
    )]
    Busy,
    /// An error on the Http Protocol level.
    #[error("HTTP request failed with status code {}. Body:\n{}", status, body)]
    Http { status: u16, body: String },
    /// Most likely either TLS errors creating the Client, or IO errors.
    #[error(transparent)]
    Other(#[from] reqwest::Error),
}

pub struct Client {
    http_client: reqwest::Client,
    pub base_url: String,
    pub api_token: String,
}

mod http {
    use super::ApiError;

    use reqwest::header::{HeaderMap, HeaderValue};
    use reqwest::{header, Client, ClientBuilder, Error, StatusCode};

    pub fn create_client(api_token: &str) -> Result<Client, Error> {
        let mut headers = HeaderMap::new();

        let mut auth_value = HeaderValue::from_str(&format!("Bearer {api_token}")).unwrap();
        // Consider marking security-sensitive headers with `set_sensitive`.
        auth_value.set_sensitive(true);
        headers.insert(header::AUTHORIZATION, auth_value);

        Ok(ClientBuilder::new().default_headers(headers).build()?)
    }

    pub async fn translate_http_error(
        response: reqwest::Response,
    ) -> Result<reqwest::Response, ApiError> {
        let status = response.status();
        if !status.is_success() {
            // Store body in a variable, so we can use it, even if it is not an Error emitted by
            // the API, but an intermediate Proxy like NGinx, so we can still forward the error
            // message.
            let body = response.text().await?;
            let translated_error = match status {
                StatusCode::TOO_MANY_REQUESTS => ApiError::TooManyRequests,
                StatusCode::SERVICE_UNAVAILABLE => ApiError::Busy,
                _ => ApiError::Http {
                    status: status.as_u16(),
                    body,
                },
            };
            Err(translated_error)
        } else {
            Ok(response)
        }
    }
}

impl Client {
    /// A new instance of an Aleph Alpha client helping you interact with the Aleph Alpha API.
    pub fn new(api_token: String) -> Result<Self, ApiError> {
        Self::new_with_base_url("https://api.aleph-alpha.com".to_owned(), api_token)
    }

    /// In production you typically would want set this to <https://api.aleph-alpha.com>. Yet
    /// you may want to use a different instances for testing.
    pub fn new_with_base_url(base_url: String, api_token: String) -> Result<Self, ApiError> {
        Ok(Self {
            http_client: http::create_client(&api_token)?,
            base_url,
            api_token,
        })
    }

    pub async fn post<I: serde::ser::Serialize, O: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        data: &I,
    ) -> Result<O, ApiError> {
        let url = format!("{base_url}{path}", base_url = self.base_url, path = path);
        let response = self.http_client.post(url).json(data).send().await?;
        let response = http::translate_http_error(response).await?;
        let response_body: O = response.json().await?;
        Ok(response_body)
    }

    pub async fn completion(&self, req: CompletionRequest) -> Result<CompletionResponse, ApiError> {
        Ok(self.post("/complete", &req).await?)
    }
}
