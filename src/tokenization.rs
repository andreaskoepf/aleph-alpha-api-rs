use serde::{Deserialize, Serialize};

#[derive(Serialize)]
pub struct TokenizationRequest {
    /// Name of the model tasked with completing the prompt. E.g. `luminous-base`.
    pub model: String,
    /// String to tokenize.
    pub prompt: String,
    /// Set this value to `true` to return text-tokens.
    pub tokens: bool,
    /// Set this value to `true to return numeric token-ids.
    pub token_ids: bool,
}

#[derive(Deserialize)]
pub struct TokenizationResponse {
    pub tokens: Option<Vec<String>>,
    pub token_ids: Option<Vec<u32>>,
}

#[derive(Serialize, Debug)]
pub struct DetokenizationRequest {
    /// Name of the model tasked with completing the prompt. E.g. `luminous-base"`.
    pub model: String,
    /// List of ids to detokenize.
    pub token_ids: Vec<u32>,
}

#[derive(Deserialize, Debug)]
pub struct DetokenizationResponse {
    pub result: String,
}
