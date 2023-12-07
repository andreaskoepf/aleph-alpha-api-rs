use super::image_processing::{from_image_path, preprocess_image, LoadImageError};
use crate::impl_builder_methods;
use base64::prelude::{Engine as _, BASE64_STANDARD};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::Path};

#[derive(Serialize, Debug)]
pub struct Prompt(Vec<Modality>);

impl Default for Prompt {
    fn default() -> Self {
        Self(vec![])
    }
}

impl Prompt {
    pub fn empty() -> Self {
        Self::default()
    }

    /// Create a prompt from a single text item.
    pub fn from_text(text: impl Into<String>) -> Self {
        Self(vec![Modality::from_text(text, None)])
    }

    pub fn from_text_with_controls(text: impl Into<String>, controls: Vec<TextControl>) -> Self {
        Self(vec![Modality::from_text(text, Some(controls))])
    }

    pub fn from_token_ids(ids: Vec<u32>, controls: Option<Vec<TokenControl>>) -> Self {
        Self(vec![Modality::from_token_ids(ids, controls)])
    }

    /// Create a multimodal prompt from a list of individual items with any modality.
    pub fn from_vec(items: Vec<Modality>) -> Self {
        Self(items)
    }
}

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct TokenControl {
    /// Index of the token, relative to the list of tokens IDs in the current prompt item.
    pub index: u32,

    /// Factor to apply to the given token in the attention matrix.
    ///
    /// - 0 <= factor < 1 => Suppress the given token
    /// - factor == 1 => identity operation, no change to attention
    /// - factor > 1 => Amplify the given token
    pub factor: f64,
}

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct TextControl {
    /// Starting character index to apply the factor to.
    start: i32,

    /// The amount of characters to apply the factor to.
    length: i32,

    /// Factor to apply to the given token in the attention matrix.
    ///
    /// - 0 <= factor < 1 => Suppress the given token
    /// - factor == 1 => identity operation, no change to attention
    /// - factor > 1 => Amplify the given token
    factor: f64,

    /// What to do if a control partially overlaps with a text token.
    ///
    /// If set to "partial", the factor will be adjusted proportionally with the amount
    /// of the token it overlaps. So a factor of 2.0 of a control that only covers 2 of
    /// 4 token characters, would be adjusted to 1.5. (It always moves closer to 1, since
    /// 1 is an identity operation for control factors.)
    ///
    /// If set to "complete", the full factor will be applied as long as the control
    /// overlaps with the token at all.
    #[serde(skip_serializing_if = "Option::is_none")]
    token_overlap: Option<String>,
}

/// Bounding box in logical coordinates. From 0 to 1. With (0,0) being the upper left corner,
/// and relative to the entire image.
///
/// Keep in mind, non-square images are center-cropped by default before going to the model.
/// (You can specify a custom cropping if you want.). Since control coordinates are relative to
/// the entire image, all or a portion of your control may be outside the "model visible area".
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct BoundingBox {
    /// x-coordinate of top left corner of the control bounding box.
    /// Must be a value between 0 and 1, where 0 is the left corner and 1 is the right corner.
    left: f64,

    /// y-coordinate of top left corner of the control bounding box
    /// Must be a value between 0 and 1, where 0 is the top pixel row and 1 is the bottom row.
    top: f64,

    /// width of the control bounding box
    /// Must be a value between 0 and 1, where 1 means the full width of the image.
    width: f64,

    /// height of the control bounding box
    /// Must be a value between 0 and 1, where 1 means the full height of the image.
    heigh: f64,
}

#[derive(Serialize, Clone, Debug)]
pub struct ImageControl {
    /// Bounding box in logical coordinates. From 0 to 1. With (0,0) being the upper left corner,
    /// and relative to the entire image.
    ///
    /// Keep in mind, non-square images are center-cropped by default before going to the model. (You
    /// can specify a custom cropping if you want.). Since control coordinates are relative to the
    /// entire image, all or a portion of your control may be outside the "model visible area".
    rect: BoundingBox,

    /// Factor to apply to the given token in the attention matrix.
    ///
    /// - 0 <= factor < 1 => Suppress the given token
    /// - factor == 1 => identity operation, no change to attention
    /// - factor > 1 => Amplify the given token
    factor: f64,

    /// What to do if a control partially overlaps with a text token.
    ///
    /// If set to "partial", the factor will be adjusted proportionally with the amount
    /// of the token it overlaps. So a factor of 2.0 of a control that only covers 2 of
    /// 4 token characters, would be adjusted to 1.5. (It always moves closer to 1, since
    /// 1 is an identity operation for control factors.)
    ///
    /// If set to "complete", the full factor will be applied as long as the control
    /// overlaps with the token at all.
    #[serde(skip_serializing_if = "Option::is_none")]
    token_overlap: Option<String>,
}

/// The prompt for models can be a combination of different modalities (Text and Image). The type of
/// modalities which are supported depend on the Model in question.
#[derive(Serialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Modality {
    /// The only type of prompt which can be used with pure language models
    Text {
        data: String,

        #[serde(skip_serializing_if = "Option::is_none")]
        controls: Option<Vec<TextControl>>,
    },
    /// An image input into the model. See [`Modality::from_image_path`].
    Image {
        /// An image send as part of a prompt to a model. The image is represented as base64.
        ///
        /// Note: The models operate on square images. All non-square images are center-cropped
        /// before going to the model, so portions of the image may not be visible.
        ///
        /// You can supply specific cropping parameters if you like, to choose a different area
        /// of the image than a center-crop. Or, you can always transform the image yourself to
        /// a square before sending it.
        data: String,

        /// x-coordinate of top left corner of cropping box in pixels
        #[serde(skip_serializing_if = "Option::is_none")]
        x: Option<i32>,

        /// y-coordinate of top left corner of cropping box in pixels
        #[serde(skip_serializing_if = "Option::is_none")]
        y: Option<i32>,

        /// Size of the cropping square in pixels
        #[serde(skip_serializing_if = "Option::is_none")]
        size: Option<i32>,

        #[serde(skip_serializing_if = "Option::is_none")]
        controls: Option<Vec<ImageControl>>,
    },
    #[serde(rename = "token_ids")]
    TokenIds {
        data: Vec<u32>,

        #[serde(skip_serializing_if = "Option::is_none")]
        controls: Option<Vec<TokenControl>>,
    },
}

impl Modality {
    /// Instantiates a text prompt
    pub fn from_text(text: impl Into<String>, controls: Option<Vec<TextControl>>) -> Self {
        Modality::Text {
            data: text.into(),
            controls,
        }
    }

    /// Instantiates a token_ids prompt
    pub fn from_token_ids(ids: Vec<u32>, controls: Option<Vec<TokenControl>>) -> Self {
        Modality::TokenIds {
            data: ids,
            controls,
        }
    }

    pub fn from_image_path(path: impl AsRef<Path>) -> Result<Self, LoadImageError> {
        let bytes = from_image_path(path.as_ref())?;
        Ok(Self::from_image_bytes(&bytes))
    }

    /// Generates an image input from the binary representation of the image.
    ///
    /// Using this constructor you must use a binary representation compatible with the API. Png is
    /// guaranteed to be supported, and all others formats are converted into it. Furthermore, the
    /// model can only look at square shaped pictures. If the picture is not square shaped it will
    /// be center cropped.
    fn from_image_bytes(image: &[u8]) -> Self {
        Modality::Image {
            data: BASE64_STANDARD.encode(image).into(),
            x: None,
            y: None,
            size: None,
            controls: None,
        }
    }

    /// Image input for model
    ///
    /// The model can only see squared pictures. Images are centercropped. You may want to use this
    /// method instead of [`Self::from_image_path`] in case you have the image in memory already
    /// and do not want to load it from a file again.
    pub fn from_image(image: &image::DynamicImage) -> Result<Self, LoadImageError> {
        let bytes = preprocess_image(image);
        Ok(Self::from_image_bytes(&bytes))
    }
}

/// Optional parameter that specifies which datacenters may process the request. You can either set the
/// parameter to "aleph-alpha" or omit it (defaulting to null).
///
/// Not setting this value, or setting it to null, gives us maximal flexibility in processing your request
/// in our own datacenters and on servers hosted with other providers. Choose this option for maximum
/// availability.
///
/// Setting it to "aleph-alpha" allows us to only process the request in our own datacenters. Choose this
/// option for maximal data privacy.
#[derive(Serialize, Debug)]
pub enum Hosting {
    #[serde(rename = "aleph-alpha")]
    AlephAlpha,
}

#[derive(Serialize, Debug, Default)]
pub struct CompletionRequest {
    /// The name of the model from the Luminous model family, e.g. `luminous-base"`.
    /// Models and their respective architectures can differ in parameter size and capabilities.
    /// The most recent version of the model is always used. The model output contains information
    /// as to the model version.
    pub model: String,

    /// Determines in which datacenters the request may be processed.
    /// You can either set the parameter to "aleph-alpha" or omit it (defaulting to None).
    ///
    /// Not setting this value, or setting it to None, gives us maximal flexibility in processing your request in our
    /// own datacenters and on servers hosted with other providers. Choose this option for maximal availability.
    ///
    /// Setting it to "aleph-alpha" allows us to only process the request in our own datacenters.
    /// Choose this option for maximal data privacy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hosting: Option<Hosting>,

    /// Prompt to complete. The modalities supported depend on `model`.
    pub prompt: Prompt,

    /// Limits the number of tokens, which are generated for the completion.
    pub maximum_tokens: u32,

    /// Generate at least this number of tokens before an end-of-text token is generated. (default: 0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum_tokens: Option<u32>,

    /// Echo the prompt in the completion. This may be especially helpful when log_probs is set to return logprobs for the
    /// prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub echo: Option<bool>,
    /// List of strings which will stop generation if they are generated. Stop sequences are
    /// helpful in structured texts. E.g.: In a question answering scenario a text may consist of
    /// lines starting with either "Question: " or "Answer: " (alternating). After producing an
    /// answer, the model will be likely to generate "Question: ". "Question: " may therefore be used
    /// as stop sequence in order not to have the model generate more questions but rather restrict
    /// text generation to the answers.

    /// A higher sampling temperature encourages the model to produce less probable outputs ("be more creative").
    /// Values are expected in a range from 0.0 to 1.0. Try high values (e.g., 0.9) for a more "creative" response and the
    /// default 0.0 for a well defined and repeatable answer. It is advised to use either temperature, top_k, or top_p, but
    /// not all three at the same time. If a combination of temperature, top_k or top_p is used, rescaling of logits with
    /// temperature will be performed first. Then top_k is applied. Top_p follows last.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Introduces random sampling for generated tokens by randomly selecting the next token from the k most likely options.
    /// A value larger than 1 encourages the model to be more creative. Set to 0.0 if repeatable output is desired. It is
    /// advised to use either temperature, top_k, or top_p, but not all three at the same time. If a combination of
    /// temperature, top_k or top_p is used, rescaling of logits with temperature will be performed first. Then top_k is
    /// applied. Top_p follows last.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    /// Introduces random sampling for generated tokens by randomly selecting the next token from the smallest possible set
    /// of tokens whose cumulative probability exceeds the probability top_p. Set to 0.0 if repeatable output is desired. It
    /// is advised to use either temperature, top_k, or top_p, but not all three at the same time. If a combination of
    /// temperature, top_k or top_p is used, rescaling of logits with temperature will be performed first. Then top_k is
    /// applied. Top_p follows last.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// The presence penalty reduces the likelihood of generating tokens that are already present in the
    /// generated text (`repetition_penalties_include_completion=true`) respectively the prompt
    /// (`repetition_penalties_include_prompt=true`).
    /// Presence penalty is independent of the number of occurrences. Increase the value to reduce the likelihood of repeating
    /// text.
    /// An operation like the following is applied: `logits[t] -> logits[t] - 1 * penalty`
    /// where `logits[t]` is the logits for any given token. Note that the formula is independent of the number of times
    /// that a token appears.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,

    /// The frequency penalty reduces the likelihood of generating tokens that are already present in the
    /// generated text (`repetition_penalties_include_completion=true`) respectively the prompt
    /// (`repetition_penalties_include_prompt=true`).
    /// If `repetition_penalties_include_prompt=True`, this also includes the tokens in the prompt.
    /// Frequency penalty is dependent on the number of occurrences of a token.
    /// An operation like the following is applied: `logits[t] -> logits[t] - count[t] * penalty`
    /// where `logits[t]` is the logits for any given token and `count[t]` is the number of times that token appears.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,

    /// Increasing the sequence penalty reduces the likelihood of reproducing token sequences that already
    /// appear in the prompt
    /// (if repetition_penalties_include_prompt is True) and prior completion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sequence_penalty: Option<f64>,

    /// Minimal number of tokens to be considered as sequence
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sequence_penalty_min_length: Option<i32>,

    /// Flag deciding whether presence penalty or frequency penalty are updated from tokens in the prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalties_include_prompt: Option<bool>,

    /// Flag deciding whether presence penalty or frequency penalty are updated from tokens in the completion
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalties_include_completion: Option<bool>,

    /// Flag deciding whether presence penalty is applied multiplicatively (True) or additively (False).
    /// This changes the formula stated for presence penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_multiplicative_presence_penalty: Option<bool>,

    /// Flag deciding whether frequency penalty is applied multiplicatively (True) or additively (False).
    /// This changes the formula stated for frequency penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_multiplicative_frequency_penalty: Option<bool>,

    /// Flag deciding whether sequence penalty is applied multiplicatively (True) or additively (False).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_multiplicative_sequence_penalty: Option<bool>,

    /// List of strings that may be generated without penalty, regardless of other penalty settings.
    /// By default, we will also include any `stop_sequences` you have set, since completion performance
    /// can be degraded if expected stop sequences are penalized.
    /// You can disable this behavior by setting `penalty_exceptions_include_stop_sequences` to `false`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub penalty_exceptions: Option<Vec<String>>,

    /// All tokens in this text will be used in addition to the already penalized tokens for repetition
    /// penalties.
    /// These consist of the already generated completion tokens and the prompt tokens, if
    /// `repetition_penalties_include_prompt` is set to `true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub penalty_bias: Option<String>,

    /// By default we include all `stop_sequences` in `penalty_exceptions`, so as not to penalise the
    /// presence of stop sequences that are present in few-shot prompts to give structure to your
    /// completions.
    ///
    /// You can set this to `false` if you do not want this behaviour.
    ///
    /// See the description of `penalty_exceptions` for more information on what `penalty_exceptions` are
    /// used for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub penalty_exceptions_include_stop_sequences: Option<bool>,

    /// If a value is given, the number of `best_of` completions will be generated on the server side. The
    /// completion with the highest log probability per token is returned. If the parameter `n` is greater
    /// than 1 more than 1 (`n`) completions will be returned. `best_of` must be strictly greater than `n`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_of: Option<i32>,

    /// The number of completions to return. If argmax sampling is used (temperature, top_k, top_p are all
    /// default) the same completions will be produced. This parameter should only be increased if random
    /// sampling is used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<i32>,

    /// Number of top log probabilities for each token generated. Log probabilities can be used in downstream
    /// tasks or to assess the model's certainty when producing tokens. No log probabilities are returned if
    /// set to None. Log probabilities of generated tokens are returned if set to 0. Log probabilities of
    /// generated tokens and top n log probabilities are returned if set to n.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_probs: Option<i32>,

    /// List of strings that will stop generation if they're generated. Stop sequences may be helpful in
    /// structured texts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,

    /// Flag indicating whether individual tokens of the completion should be returned (True) or whether
    /// solely the generated text (i.e. the completion) is sufficient (False).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<bool>,

    /// Setting this parameter to true forces the raw completion of the model to be returned.
    /// For some models, we may optimize the completion that was generated by the model and
    /// return the optimized completion in the completion field of the CompletionResponse.
    /// The raw completion, if returned, will contain the un-optimized completion.
    /// Setting tokens to true or log_probs to any value will also trigger the raw completion
    /// to be returned.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_completion: Option<bool>,

    /// We continually research optimal ways to work with our models. By default, we apply these
    /// optimizations to both your prompt and completion for you.
    /// Our goal is to improve your results while using our API. But you can always pass
    /// `disable_optimizations: true` and we will leave your prompt and completion untouched.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_optimizations: Option<bool>,

    /// Bias the completion to only generate options within this list;
    /// all other tokens are disregarded at sampling
    ///
    /// Note that strings in the inclusion list must not be prefixes
    /// of strings in the exclusion list and vice versa
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_bias_inclusion: Option<Vec<String>>,

    /// Only consider the first token for the completion_bias_inclusion
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_bias_inclusion_first_token_only: Option<bool>,

    /// Bias the completion to NOT generate options within this list;
    /// all other tokens are unaffected in sampling
    ///
    /// Note that strings in the inclusion list must not be prefixes
    /// of strings in the exclusion list and vice versa
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_bias_exclusion: Option<Vec<String>>,

    /// Only consider the first token for the completion_bias_exclusion
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_bias_exclusion_first_token_only: Option<bool>,

    /// If set to `null`, attention control parameters only apply to those tokens that have
    /// explicitly been set in the request.
    /// If set to a non-null value, we apply the control parameters to similar tokens as well.
    /// Controls that have been applied to one token will then be applied to all other tokens
    /// that have at least the similarity score defined by this parameter.
    /// The similarity score is the cosine similarity of token embeddings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contextual_control_threshold: Option<f64>,

    /// `true`: apply controls on prompt items by adding the `log(control_factor)` to attention scores.
    /// `false`: apply controls on prompt items by
    /// `(attention_scores - -attention_scores.min(-1)) * control_factor`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control_log_additive: Option<bool>,

    /// The logit bias allows to influence the likelihood of generating tokens. A dictionary mapping token
    /// ids (int) to a bias (float) can be provided. Such bias is added to the logits as generated by the
    /// model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<i32, f32>>,
}

impl CompletionRequest {
    pub fn new(model: String, prompt: Prompt, maximum_tokens: u32) -> Self {
        Self {
            model,
            prompt,
            maximum_tokens,
            ..Self::default()
        }
    }
}

impl_builder_methods!(
    CompletionRequest,
    minimum_tokens: u32,
    echo: bool,
    temperature: f64,
    top_k: u32,
    top_p: f64,
    presence_penalty: f64,
    frequency_penalty: f64,
    sequence_penalty: f64,
    sequence_penalty_min_length: i32,
    repetition_penalties_include_prompt: bool,
    repetition_penalties_include_completion: bool,
    use_multiplicative_presence_penalty: bool,
    use_multiplicative_frequency_penalty: bool,
    use_multiplicative_sequence_penalty: bool,
    penalty_exceptions: Vec<String>,
    penalty_bias: String,
    penalty_exceptions_include_stop_sequences: bool,
    best_of: i32,
    n: i32,
    log_probs: i32,
    stop_sequences: Vec<String>,
    tokens: bool,
    raw_completion: bool,
    disable_optimizations: bool,
    completion_bias_inclusion: Vec<String>,
    completion_bias_inclusion_first_token_only: bool,
    completion_bias_exclusion: Vec<String>,
    completion_bias_exclusion_first_token_only: bool,
    contextual_control_threshold: f64,
    control_log_additive: bool,
    logit_bias: HashMap<i32, f32>
);

#[derive(Deserialize, Debug)]
pub struct CompletionResponse {
    /// model name and version (if any) of the used model for inference
    pub model_version: String,
    /// list of completions; may contain only one entry if no more are requested (see parameter n)
    pub completions: Vec<CompletionOutput>,
}

impl CompletionResponse {
    /// The best completion in the answer.
    pub fn best(&self) -> &CompletionOutput {
        self.completions
            .first()
            .expect("Response is assumed to always have at least one completion")
    }

    /// Text of the best completion.
    pub fn best_text(&self) -> &str {
        &self.best().completion
    }
}

#[derive(Deserialize, Debug)]
pub struct CompletionOutput {
    pub completion: String,
    pub finish_reason: String,
}
