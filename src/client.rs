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
use bytes::Bytes;
use tokenizers::Tokenizer;

pub struct Client {
    http_client: reqwest::Client,
    pub base_url: String,
    pub api_token: String,
}

pub const ALEPH_ALPHA_API_BASE_URL: &str = "https://api.aleph-alpha.com";

impl Client {
    /// A new instance of an Aleph Alpha client helping you interact with the Aleph Alpha API.
    pub fn new(api_token: String) -> Result<Self, ApiError> {
        Self::new_with_base_url(ALEPH_ALPHA_API_BASE_URL.to_owned(), api_token)
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
    /// Example usage:
    /// ```
    ///use aleph_alpha_api::{error::ApiError, Client, CompletionRequest, LUMINOUS_BASE};
    ///
    ///const AA_API_TOKEN: &str = "<YOUR_AA_API_TOKEN>";
    ///
    ///async fn print_completion() -> Result<(), ApiError> {
    ///    let client = Client::new(AA_API_TOKEN.to_owned())?;
    ///
    ///    let request =
    ///        CompletionRequest::from_text(LUMINOUS_BASE.to_owned(), "An apple a day".to_owned(), 10)
    ///            .temperature(0.8)
    ///            .top_k(50)
    ///            .top_p(0.95)
    ///            .best_of(2)
    ///            .minimum_tokens(2);
    ///
    ///    let response = client.completion(&request, Some(true)).await?;
    ///
    ///    println!("An apple a day{}", response.best_text());
    ///
    ///    Ok(())
    ///}
    /// ```
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
}
