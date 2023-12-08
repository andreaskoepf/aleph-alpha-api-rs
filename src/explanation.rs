use super::completion::{BoundingBox, Hosting, Prompt};
use crate::impl_builder_methods;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Debug, Default)]
#[serde(rename_all = "snake_case")]
pub enum Postprocessing {
    /// Apply no postprocessing.
    #[default]
    None,
    /// Return the absolute value of each value.
    Absolute,
    /// Square each value
    Square,
}

#[derive(Serialize, Debug, Default)]
#[serde(rename_all = "snake_case")]
pub enum PromptGranularityType {
    #[default]
    Token,
    Word,
    Sentence,
    Paragraph,
    Custom,
}

#[derive(Serialize, Debug)]
pub struct PromptGranularity {
    /// At which granularity should the target be explained in terms of the prompt.
    /// If you choose, for example, "sentence" then we report the importance score of each
    /// sentence in the prompt towards generating the target output.
    ///
    /// If you do not choose a granularity then we will try to find the granularity that
    /// brings you closest to around 30 explanations. For large documents, this would likely
    /// be sentences. For short prompts this might be individual words or even tokens.
    ///
    /// If you choose a custom granularity then you must provide a custom delimiter. We then
    /// split your prompt by that delimiter. This might be helpful if you are using few-shot
    /// prompts that contain stop sequences.
    ///
    /// For image prompt items, the granularities determine into how many tiles we divide
    /// the image for the explanation.
    /// "token" -> 12x12
    /// "word" -> 6x6
    /// "sentence" -> 3x3
    /// "paragraph" -> 1
    #[serde(rename = "type")]
    granularity_type: PromptGranularityType,

    /// A delimiter string to split the prompt on if "custom" granularity is chosen.
    delimiter: String,
}

impl Default for PromptGranularity {
    fn default() -> Self {
        Self {
            granularity_type: PromptGranularityType::default(),
            delimiter: String::new(),
        }
    }
}

/// How many explanations should be returned in the output.
#[derive(Serialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum TargetGranularity {
    /// Return one explanation for the entire target. Helpful in many cases to determine which parts of the prompt contribute overall to the given completion.
    Complete,

    /// Return one explanation for each token in the target.
    Token,
}

#[derive(Serialize, Debug, Default)]
#[serde(rename_all = "snake_case")]
pub enum ControlTokenOverlap {
    #[default]
    Partial,
    Complete,
}

#[derive(Serialize, Debug, Default)]
pub struct ExplanationRequest {
    /// Name of the model to use.
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

    pub prompt: Prompt,

    /// The completion string to be explained based on model probabilities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,

    /// Factor to apply to the given token in the attention matrix.
    ///
    /// - 0 <= factor < 1 => Suppress the given token
    /// - factor == 1 => identity operation, no change to attention
    /// - factor > 1 => Amplify the given token
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control_factor: Option<f64>,

    /// If set to `null`, attention control parameters only apply to those tokens that have explicitly been set in the request.
    /// If set to a non-null value, we apply the control parameters to similar tokens as well.
    /// Controls that have been applied to one token will then be applied to all other tokens that have at least the similarity score defined by this parameter.
    /// The similarity score is the cosine similarity of token embeddings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contextual_control_threshold: Option<f64>,

    /// `true`: apply controls on prompt items by adding the `log(control_factor)` to attention scores.
    /// `false`: apply controls on prompt items by `(attention_scores - -attention_scores.min(-1)) * control_factor`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control_log_additive: Option<bool>,

    /// Optionally apply postprocessing to the difference in cross entropy scores for each token.
    /// "none": Apply no postprocessing.
    /// "absolute": Return the absolute value of each value.
    /// "square": Square each value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub postprocessing: Option<Postprocessing>,

    /// Return normalized scores. Minimum score becomes 0 and maximum score becomes 1. Applied after any postprocessing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalize: Option<bool>,

    /// At which granularity should the target be explained in terms of the prompt.
    /// If you choose, for example, "sentence" then we report the importance score of each
    /// sentence in the prompt towards generating the target output.
    ///
    /// If you do not choose a granularity then we will try to find the granularity that
    /// brings you closest to around 30 explanations. For large documents, this would likely
    /// be sentences. For short prompts this might be individual words or even tokens.
    ///
    /// If you choose a custom granularity then you must provide a custom delimiter. We then
    /// split your prompt by that delimiter. This might be helpful if you are using few-shot
    /// prompts that contain stop sequences.
    ///
    /// For image prompt items, the granularities determine into how many tiles we divide
    /// the image for the explanation.
    /// Token -> 12x12
    /// Word -> 6x6
    /// Sentence -> 3x3
    /// Paragraph -> 1
    pub prompt_granularity: Option<PromptGranularity>,

    /// How many explanations should be returned in the output.
    ///
    /// Complete -> Return one explanation for the entire target. Helpful in many cases to determine which parts of the prompt contribute overall to the given completion.
    /// Token -> Return one explanation for each token in the target.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_granularity: Option<TargetGranularity>,

    /// What to do if a control partially overlaps with a text or image token.
    ///
    /// If set to "partial", the factor will be adjusted proportionally with the amount
    /// of the token it overlaps. So a factor of 2.0 of a control that only covers 2 of
    /// 4 token characters, would be adjusted to 1.5. (It always moves closer to 1, since
    /// 1 is an identity operation for control factors.)
    ///
    /// If set to "complete", the full factor will be applied as long as the control
    /// overlaps with the token at all.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control_token_overlap: Option<ControlTokenOverlap>,
}

impl_builder_methods!(
    ExplanationRequest,
    hosting: Hosting,
    target: String,
    control_factor: f64,
    contextual_control_threshold: f64,
    control_log_additive: bool,
    postprocessing: Postprocessing,
    normalize: bool,
    prompt_granularity: PromptGranularity,
    target_granularity: TargetGranularity,
    control_token_overlap: ControlTokenOverlap
);

#[derive(Deserialize, Debug)]
pub struct ScoredSegment {
    pub start: i32,
    pub length: i32,
    pub score: f32,
}

#[derive(Deserialize, Debug)]
pub struct ScoredRect {
    pub rect: BoundingBox,
    pub score: f32,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ItemImportance {
    /// Explains the importance of a request prompt item of type "token_ids".
    /// Will contain one floating point importance value for each token in the same order as in
    /// the original prompt.
    TokenIds { scores: Vec<f32> },

    /// Explains the importance of text in the target string that came before the currently to-be-explained target token. The amount of items in the "scores" array depends on the granularity setting.
    /// Each score object contains an inclusive start character and a length of the substring plus a floating point score value.
    Target { scores: Vec<ScoredSegment> },

    /// Explains the importance of a text prompt item.
    /// The amount of items in the "scores" array depends on the granularity setting.
    /// Each score object contains an inclusive start character and a length of the substring plus a floating point score value.
    Text { scores: Vec<ScoredSegment> },

    /// Explains the importance of an image prompt item.
    /// The amount of items in the "scores" array depends on the granularity setting.
    /// Each score object contains the top-left corner of a rectangular area in the image prompt.
    /// The coordinates are all between 0 and 1 in terms of the total image size
    Image { scores: Vec<ScoredRect> },
}

#[derive(Deserialize, Debug)]
pub struct ExplanationItem {
    /// The string representation of the target token which is being explained
    pub target: String,

    /// Contains one item for each prompt item (in order), and the last item refers to the target.
    pub items: Vec<ItemImportance>,
}

/// The top-level response data structure that will be returned from an explanation request.
#[derive(Deserialize, Debug)]
pub struct ExplanationResponse {
    pub model_version: String,

    /// This array will contain one explanation object for each token in the target string.
    pub explanations: Vec<ExplanationItem>,
}
