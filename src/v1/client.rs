use super::api_tokens::{CreateApiTokenRequest, CreateApiTokenResponse, ListApiTokensResponse};
use super::completion::{CompletionRequest, CompletionResponse};
use super::embedding::{
    BatchSemanticEmbeddingRequest, BatchSemanticEmbeddingResponse, EmbeddingRequest,
    EmbeddingResponse, SemanticEmbeddingRequest, SemanticEmbeddingResponse,
};
use super::error::ApiError;
use super::evaluate::{EvaluationRequest, EvaluationResponse};
use super::explanation::{ExplanationRequest, ExplanationResponse};
use super::http;
use super::tokenization::{
    DetokenizationRequest, DetokenizationResponse, TokenizationRequest, TokenizationResponse,
};
use super::users::{UserChange, UserDetail};
use bytes::Bytes;
use tokenizers::Tokenizer;

pub struct Client {
    http_client: reqwest::Client,
    pub base_url: String,
    pub api_token: String,
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
        query: Option<Vec<(String, String)>>,
    ) -> Result<O, ApiError> {
        use reqwest::header::{ACCEPT, CONTENT_TYPE};

        let url = format!("{base_url}{path}", base_url = self.base_url, path = path);
        let mut request = self.http_client.post(url);

        if let Some(q) = query {
            request = request.query(&q);
        }

        let request = request
            .header(CONTENT_TYPE, "application/json")
            .header(ACCEPT, "application/json")
            .json(data);

        let response = request.send().await?;
        let response = http::translate_http_error(response).await?;
        let response_body: O = response.json().await?;
        Ok(response_body)
    }

    pub async fn post_nice<I: serde::ser::Serialize, O: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        data: &I,
        nice: Option<bool>,
    ) -> Result<O, ApiError> {
        let query = if let Some(be_nice) = nice {
            Some(vec![("nice".to_owned(), be_nice.to_string())])
        } else {
            None
        };
        Ok(self.post(path, data, query).await?)
    }

    pub async fn get<O: serde::de::DeserializeOwned>(&self, path: &str) -> Result<O, ApiError> {
        let response = http::get(&self.http_client, &self.base_url, path, None).await?;
        let response_body = response.json().await?;
        Ok(response_body)
    }

    pub async fn get_string(&self, path: &str) -> Result<String, ApiError> {
        let response = http::get(&self.http_client, &self.base_url, path, None).await?;
        let response_body = response.text().await?;
        Ok(response_body)
    }

    pub async fn get_binary(&self, path: &str) -> Result<Bytes, ApiError> {
        let response = http::get(&self.http_client, &self.base_url, path, None).await?;
        let response_body = response.bytes().await?;
        Ok(response_body)
    }

    /// Will complete a prompt using a specific model.
    pub async fn completion(
        &self,
        req: &CompletionRequest,
        nice: Option<bool>,
    ) -> Result<CompletionResponse, ApiError> {
        Ok(self.post_nice("/complete", req, nice).await?)
    }

    /// Evaluates the model's likelihood to produce a completion given a prompt.
    pub async fn evaluate(
        &self,
        req: &EvaluationRequest,
        nice: Option<bool>,
    ) -> Result<EvaluationResponse, ApiError> {
        Ok(self.post_nice("/evaluate", req, nice).await?)
    }

    /// Better understand the source of a completion, specifically on how much each section of a prompt impacts each token of the completion.
    pub async fn explain(
        &self,
        req: &ExplanationRequest,
        nice: Option<bool>,
    ) -> Result<ExplanationResponse, ApiError> {
        Ok(self.post_nice("/explain", req, nice).await?)
    }

    /// Embeds a text using a specific model. Resulting vectors that can be used for downstream tasks (e.g. semantic similarity) and models (e.g. classifiers).
    pub async fn embed(
        &self,
        req: &EmbeddingRequest,
        nice: Option<bool>,
    ) -> Result<EmbeddingResponse, ApiError> {
        Ok(self.post_nice("/embed", req, nice).await?)
    }

    /// Embeds a prompt using a specific model and semantic embedding method. Resulting vectors that can be used for downstream tasks (e.g. semantic similarity) and models (e.g. classifiers). To obtain a valid model,
    pub async fn semantic_embed(
        &self,
        req: &SemanticEmbeddingRequest,
        nice: Option<bool>,
    ) -> Result<SemanticEmbeddingResponse, ApiError> {
        Ok(self.post_nice("/semantic_embed", req, nice).await?)
    }

    /// Embeds multiple prompts using a specific model and semantic embedding method. Resulting vectors that can be used for downstream tasks (e.g. semantic similarity) and models (e.g. classifiers).
    pub async fn batch_semantic_embed(
        &self,
        req: &BatchSemanticEmbeddingRequest,
        nice: Option<bool>,
    ) -> Result<BatchSemanticEmbeddingResponse, ApiError> {
        Ok(self.post_nice("/batch_semantic_embed", req, nice).await?)
    }

    /// Tokenize a prompt for a specific model.
    pub async fn tokenize(
        &self,
        req: &TokenizationRequest,
    ) -> Result<TokenizationResponse, ApiError> {
        Ok(self.post("/tokenize", req, None).await?)
    }

    /// Detokenize a list of tokens into a string.
    pub async fn detokenize(
        &self,
        req: &DetokenizationRequest,
    ) -> Result<DetokenizationResponse, ApiError> {
        Ok(self.post("/detokenize", req, None).await?)
    }

    pub async fn get_tokenizer_binary(&self, model: &str) -> Result<Bytes, ApiError> {
        let path = format!("/models/{model}/tokenizer");
        let vocabulary = self.get_binary(&path).await?;
        Ok(vocabulary)
    }

    pub async fn get_tokenizer(&self, model: &str) -> Result<Tokenizer, ApiError> {
        let vocabulary = self.get_tokenizer_binary(model).await?;
        let tokenizer = Tokenizer::from_bytes(vocabulary)?;
        Ok(tokenizer)
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
        Ok(self.post("/users/me/tokens", req, None).await?)
    }

    /// Delete an API token
    pub async fn delete_api_token(&self, token_id: i32) -> Result<(), ApiError> {
        let path = format!("/users/me/tokens/{token_id}");
        http::delete(&self.http_client, &self.base_url, &path).await?;
        Ok(())
    }

    /// Get settings for own user
    pub async fn get_user_settings(&self) -> Result<UserDetail, ApiError> {
        Ok(self.get("/users/me").await?)
    }

    /// Change settings for own user
    pub async fn change_user_settings(
        &self,
        settings: &UserChange,
    ) -> Result<UserDetail, ApiError> {
        Ok(self.post("/users/me", settings, None).await?)
    }
}
