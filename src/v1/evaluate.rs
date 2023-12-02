use super::completion::{Hosting, Prompt};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Debug, Default)]
pub struct EvaluationRequest {
    pub model: String,

    /// Base prompt to for the evaluation.
    pub prompt: Prompt,

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

    /// The completion that you would expect to be completed. Unconditional completion can be used with an
    /// empty string (default). The prompt may contain a zero shot or few shot task.
    pub completion_expected: String,

    /// If set to `None`, attention control parameters only apply to those tokens that have explicitly been set
    /// in the request. If set to a non-null value, we apply the control parameters to similar tokens as
    /// well. Controls that have been applied to one token will then be applied to all other tokens that have
    /// at least the similarity score defined by this parameter. The similarity score is the cosine
    /// similarity of token embeddings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contextual_control_threshold: Option<f64>,

    /// Default value: true
    /// true: apply controls on prompt items by adding the `log(control_factor)`` to attention scores.
    /// false: apply controls on prompt items by `(attention_scores - -attention_scores.min(-1)) * control_factor`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control_log_additive: Option<bool>,
}

impl EvaluationRequest {
    pub fn from_text(
        model: impl Into<String>,
        prompt: impl Into<String>,
        completion_expected: impl Into<String>,
    ) -> Self {
        Self {
            model: model.into(),
            prompt: Prompt::from_text(prompt),
            completion_expected: completion_expected.into(),
            ..Self::default()
        }
    }
}

#[derive(Deserialize, Debug)]
pub struct EvaluationResponse {
    /// model name and version (if any) of the used model for inference
    pub model_version: String,

    /// object with result metrics of the evaluation
    pub result: EvaluationResult,
}

#[derive(Deserialize, Debug)]
pub struct EvaluationResult {
    /// log probability of producing the expected completion given the prompt. This metric refers to all tokens and is therefore dependent on the used tokenizer. It cannot be directly compared among models with different tokenizers.
    pub log_probability: Option<f64>,

    /// log perplexity associated with the expected completion given the prompt. This metric refers to all tokens and is therefore dependent on the used tokenizer. It cannot be directly compared among models with different tokenizers.
    pub log_perplexity: Option<f64>,

    /// log perplexity associated with the expected completion given the prompt normalized for the number of tokens. This metric computes an average per token and is therefore dependent on the used tokenizer. It cannot be directly compared among models with different tokenizers.
    pub log_perplexity_per_token: Option<f64>,

    /// log perplexity associated with the expected completion given the prompt normalized for the number of characters. This metric is independent of any tokenizer. It can be directly compared among models with different tokenizers.
    pub log_perplexity_per_character: Option<f64>,

    /// Flag indicating whether a greedy completion would have produced the expected completion.
    pub correct_greedy: Option<bool>,

    /// Number of tokens in the expected completion.
    pub token_count: Option<i32>,

    /// Number of characters in the expected completion.
    pub character_count: Option<i32>,

    /// argmax completion given the input consisting of prompt and expected completion. This may be used as an indicator of what the model would have produced. As only one single forward is performed an incoherent text could be produced especially for long expected completions.
    pub completion: Option<String>,
}
