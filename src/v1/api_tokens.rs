use serde::{Deserialize, Serialize};

#[derive(Deserialize, Debug)]
pub struct ApiTokenMetadata {
    /// A simple description that was supplied when creating the token
    pub description: Option<String>,
    /// The token ID to use when calling other endpoints
    pub token_id: i32,
}

pub type ListApiTokensResponse = Vec<ApiTokenMetadata>;

#[derive(Serialize, Debug)]
pub struct CreateApiTokenRequest {
    /// a simple description to remember the token by
    pub description: String,
}

#[derive(Deserialize, Debug)]
pub struct CreateApiTokenResponse {
    pub metadata: ApiTokenMetadata,
    /// the API token that can be used in the Authorization header
    pub token: String,
}
