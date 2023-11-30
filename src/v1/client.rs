use super::api_tokens::{CreateApiTokenRequest, CreateApiTokenResponse, ListApiTokensResponse};
use super::completion::{CompletionRequest, CompletionResponse};
use super::tokenization::{
    DetokenizationRequest, DetokenizationResponse, TokenizationRequest, TokenizationResponse,
};

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

    pub async fn get(
        client: &reqwest::Client,
        base_url: &str,
        path: &str,
    ) -> Result<reqwest::Response, ApiError> {
        let url = format!("{base_url}{path}");
        let response = client.get(url).send().await?;
        translate_http_error(response).await
    }

    pub async fn delete(
        client: &reqwest::Client,
        base_url: &str,
        path: &str,
    ) -> Result<reqwest::Response, ApiError> {
        let url = format!("{base_url}{path}");
        let response = client.delete(url).send().await?;
        translate_http_error(response).await
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

    pub async fn get<O: serde::de::DeserializeOwned>(&self, path: &str) -> Result<O, ApiError> {
        let response = http::get(&self.http_client, &self.base_url, path).await?;
        let response_body = response.json().await?;
        Ok(response_body)
    }

    pub async fn get_string(&self, path: &str) -> Result<String, ApiError> {
        let response = http::get(&self.http_client, &self.base_url, path).await?;
        let response_body = response.text().await?;
        Ok(response_body)
    }

    pub async fn completion(
        &self,
        req: &CompletionRequest,
    ) -> Result<CompletionResponse, ApiError> {
        Ok(self.post("/complete", req).await?)
    }

    pub async fn tokenize(
        &self,
        req: &TokenizationRequest,
    ) -> Result<TokenizationResponse, ApiError> {
        Ok(self.post("/tokenize", req).await?)
    }

    pub async fn detokenize(
        &self,
        req: &DetokenizationRequest,
    ) -> Result<DetokenizationResponse, ApiError> {
        Ok(self.post("/detokenize", req).await?)
    }

    /// Will return the version number of the API that is deployed to this environment.
    pub async fn get_version(&self) -> Result<String, ApiError> {
        Ok(self.get_string("/version").await?)
    }

    /// Will return a list of API tokens that are registered for this user (only token metadata is returned, not the actual tokens)
    pub async fn list_api_tokens(&self) -> Result<ListApiTokensResponse, ApiError> {
        Ok(self.get("/users/me/tokens").await?)
    }

    /// Create a new token to authenticate against the API with (the actual API token is only returned when calling this endpoint)
    pub async fn create_api_token(
        &self,
        req: &CreateApiTokenRequest,
    ) -> Result<CreateApiTokenResponse, ApiError> {
        Ok(self.post("/users/me/tokens", req).await?)
    }

    pub async fn delete_api_token(&self, token_id: i32) -> Result<(), ApiError> {
        let path = format!("/users/me/tokens/{token_id}");
        http::delete(&self.http_client, &self.base_url, &path).await?;
        Ok(())
    }
}