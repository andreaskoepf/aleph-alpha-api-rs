use super::completion::{Hosting, Prompt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Debug, Default)]
pub struct EmbeddingRequest {
    /// Name of model to use. A model name refers to a model architecture (number of parameters among others). Always the latest version of model is used. The model output contains information as to the model version.
    pub model: String,

    /// Possible values: [aleph-alpha, None]
    /// Optional parameter that specifies which datacenters may process the request. You can either set the
    /// parameter to "aleph-alpha" or omit it (defaulting to null).
    /// Not setting this value, or setting it to None, gives us maximal flexibility in processing your
    /// request in our own datacenters and on servers hosted with other providers. Choose this option for
    /// maximum availability.
    /// Setting it to "aleph-alpha" allows us to only process the request in our own datacenters. Choose this
    /// option for maximal data privacy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hosting: Option<Hosting>,

    /// This field is used to send prompts to the model. A prompt can either be a text prompt or a multimodal prompt. A text prompt is a string of text. A multimodal prompt is an array of prompt items. It can be a combination of text, images, and token ID arrays.
    /// In the case of a multimodal prompt, the prompt items will be concatenated and a single prompt will be used for the model.
    /// Tokenization:
    /// Token ID arrays are used as as-is.
    /// Text prompt items are tokenized using the tokenizers specific to the model.
    /// Each image is converted into 144 tokens.
    pub prompt: Prompt,

    /// A list of layer indices from which to return embeddings.
    /// - Index 0 corresponds to the word embeddings used as input to the first transformer layer
    /// - Index 1 corresponds to the hidden state as output by the first transformer layer, index 2 to the output of the second layer etc.
    /// - Index -1 corresponds to the last transformer layer (not the language modelling head), index -2 to the second last
    pub layers: Vec<i32>,

    /// Flag indicating whether the tokenized prompt is to be returned (True) or not (False)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<bool>,

    /// Pooling operation to use. Pooling operations include:
    /// - "mean": Aggregate token embeddings across the sequence dimension using an average.
    /// - "weighted_mean": Position weighted mean across sequence dimension with latter tokens having a higher weight.
    /// - "max": Aggregate token embeddings across the sequence dimension using a maximum.
    /// - "last_token": Use the last token.
    /// - "abs_max": Aggregate token embeddings across the sequence dimension using a maximum of absolute values.
    pub pooling: Vec<String>,

    /// Explicitly set embedding type to be passed to the model. This parameter was created to allow for semantic_embed embeddings and will be deprecated. Please use the semantic_embed-endpoint instead.
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    // type is a reserved word in Rust
    pub embedding_type: Option<String>,

    /// Return normalized embeddings. This can be used to save on additional compute when applying a cosine similarity metric.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalize: Option<bool>,

    /// If set to `None`, attention control parameters only apply to those tokens that have explicitly been set
    /// in the request. If set to a non-null value, we apply the control parameters to similar tokens as
    /// well. Controls that have been applied to one token will then be applied to all other tokens that have
    /// at least the similarity score defined by this parameter. The similarity score is the cosine
    /// similarity of token embeddings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contextual_control_threshold: Option<f64>,

    /// `true`: apply controls on prompt items by adding the `log(control_factor)` to attention scores.
    /// `false`: apply controls on prompt items by `(attention_scores - -attention_scores.min(-1)) * control_factor`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control_log_additive: Option<bool>,
}

impl EmbeddingRequest {
    pub fn from_text(
        model: impl Into<String>,
        prompt: impl Into<String>,
        layer: i32,
        pooling: impl Into<String>,
        normalize: bool,
    ) -> Self {
        Self {
            model: model.into(),
            prompt: Prompt::from_text(prompt),
            layers: vec![layer.into()],
            pooling: vec![pooling.into()],
            normalize: Some(normalize),
            ..Self::default()
        }
    }
}

type Embedding = Vec<f32>;
type PoolingEmbeddings = HashMap<String, Embedding>;
type LayerEmbedings = HashMap<String, PoolingEmbeddings>;

#[derive(Deserialize, Debug)]
pub struct EmbeddingResponse {
    /// model name and version (if any) of the used model for inference
    pub model_version: String,

    /// embeddings:
    /// - pooling: a dict with layer names as keys and and pooling output as values. A pooling output is a dict with pooling operation as key and a pooled embedding (list of floats) as values
    pub embeddings: LayerEmbedings,

    pub tokens: Option<Vec<String>>,
}

/// Type of embedding representation to embed the prompt with.
///
/// `"symmetric"`: Symmetric embeddings assume that the text to be compared is interchangeable. Usage examples for symmetric embeddings are clustering, classification, anomaly detection or visualisation tasks. "symmetric" embeddings should be compared with other "symmetric" embeddings.
///
/// `"document"` and `"query"`: Asymmetric embeddings assume that there is a difference between queries and documents. They are used together in use cases such as search where you want to compare shorter queries against larger documents.
///
/// `"query"`-embeddings are optimized for shorter texts, such as questions or keywords.
///
/// `"document"`-embeddings are optimized for larger pieces of text to compare queries against.
#[derive(Serialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingRepresentation {
    Symmetric,
    Document,
    Query,
}

impl Default for EmbeddingRepresentation {
    fn default() -> Self {
        EmbeddingRepresentation::Symmetric
    }
}

/// Embeds a prompt using a specific model and semantic embedding method. Resulting vectors that can be used for downstream tasks (e.g. semantic similarity) and models (e.g. classifiers).
#[derive(Serialize, Debug, Default)]
pub struct SemanticEmbeddingRequest {
    /// Name of the model to use. A model name refers to a model's architecture (number of parameters among others). The most recent version of the model is always used. The model output contains information as to the model version. To create semantic embeddings, please use `luminous-base`.
    pub model: String,

    /// Possible values: [aleph-alpha, None]
    /// Optional parameter that specifies which datacenters may process the request. You can either set the
    /// parameter to "aleph-alpha" or omit it (defaulting to null).
    /// Not setting this value, or setting it to None, gives us maximal flexibility in processing your
    /// request in our own datacenters and on servers hosted with other providers. Choose this option for
    /// maximum availability.
    /// Setting it to "aleph-alpha" allows us to only process the request in our own datacenters. Choose this
    /// option for maximal data privacy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hosting: Option<Hosting>,

    /// This field is used to send prompts to the model. A prompt can either be a text prompt or a multimodal prompt. A text prompt is a string of text. A multimodal prompt is an array of prompt items. It can be a combination of text, images, and token ID arrays.
    /// In the case of a multimodal prompt, the prompt items will be concatenated and a single prompt will be used for the model.
    /// Tokenization:
    /// Token ID arrays are used as as-is.
    /// Text prompt items are tokenized using the tokenizers specific to the model.
    /// Each image is converted into 144 tokens.
    pub prompt: Prompt,

    /// Type of embedding representation to embed the prompt with.
    pub representation: EmbeddingRepresentation,

    /// The default behavior is to return the full embedding with 5120 dimensions. With this parameter you can compress the returned embedding to 128 dimensions.
    /// The compression is expected to result in a small drop in accuracy performance (4-6%), with the benefit of being much smaller, which makes comparing these embeddings much faster for use cases where speed is critical.
    /// With the compressed embedding can also perform better if you are embedding really short texts or documents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compress_to_size: Option<i32>,

    /// Return normalized embeddings. This can be used to save on additional compute when applying a cosine similarity metric.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalize: Option<bool>,

    /// If set to `null`, attention control parameters only apply to those tokens that have explicitly been set in the request.
    /// If set to a non-null value, we apply the control parameters to similar tokens as well.
    /// Controls that have been applied to one token will then be applied to all other tokens
    /// that have at least the similarity score defined by this parameter.
    /// The similarity score is the cosine similarity of token embeddings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contextual_control_threshold: Option<f64>,

    /// `true`: apply controls on prompt items by adding the `log(control_factor)` to attention scores.
    /// `false`: apply controls on prompt items by `(attention_scores - -attention_scores.min(-1)) * control_factor`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control_log_additive: Option<bool>,
}

#[derive(Deserialize, Debug)]
pub struct SemanticEmbeddingResponse {
    /// model name and version (if any) of the used model for inference
    pub model_version: String,

    /// A list of floats that can be used to compare against other embeddings.
    pub embedding: Embedding,
}

#[derive(Serialize, Debug, Default)]
pub struct BatchSemanticEmbeddingRequest {
    /// Name of the model to use. A model name refers to a model's architecture (number of parameters among others). The most recent version of the model is always used. The model output contains information as to the model version. To create semantic embeddings, please use `luminous-base`.
    pub model: String,

    /// Possible values: [aleph-alpha, None]
    /// Optional parameter that specifies which datacenters may process the request. You can either set the
    /// parameter to "aleph-alpha" or omit it (defaulting to null).
    /// Not setting this value, or setting it to None, gives us maximal flexibility in processing your
    /// request in our own datacenters and on servers hosted with other providers. Choose this option for
    /// maximum availability.
    /// Setting it to "aleph-alpha" allows us to only process the request in our own datacenters. Choose this
    /// option for maximal data privacy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hosting: Option<Hosting>,

    /// This field is used to send prompts to the model. A prompt can either be a text prompt or a multimodal prompt. A text prompt is a string of text. A multimodal prompt is an array of prompt items. It can be a combination of text, images, and token ID arrays.
    pub prompts: Vec<Prompt>,

    /// Type of embedding representation to embed the prompt with.
    pub representation: EmbeddingRepresentation,

    /// The default behavior is to return the full embedding with 5120 dimensions. With this parameter you can compress the returned embedding to 128 dimensions.
    /// The compression is expected to result in a small drop in accuracy performance (4-6%), with the benefit of being much smaller, which makes comparing these embeddings much faster for use cases where speed is critical.
    /// With the compressed embedding can also perform better if you are embedding really short texts or documents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compress_to_size: Option<i32>,

    /// Return normalized embeddings. This can be used to save on additional compute when applying a cosine similarity metric.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalize: Option<bool>,

    /// If set to `null`, attention control parameters only apply to those tokens that have explicitly been set in the request.
    /// If set to a non-null value, we apply the control parameters to similar tokens as well.
    /// Controls that have been applied to one token will then be applied to all other tokens
    /// that have at least the similarity score defined by this parameter.
    /// The similarity score is the cosine similarity of token embeddings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contextual_control_threshold: Option<f64>,

    /// `true`: apply controls on prompt items by adding the `log(control_factor)` to attention scores.
    /// `false`: apply controls on prompt items by `(attention_scores - -attention_scores.min(-1)) * control_factor`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control_log_additive: Option<bool>,
}

#[derive(Deserialize, Debug)]
pub struct BatchSemanticEmbeddingResponse {
    /// model name and version (if any) of the used model for inference
    pub model_version: String,

    /// Vector of embeddings (one fore each prompt)
    pub embeddings: Vec<Embedding>,
}
