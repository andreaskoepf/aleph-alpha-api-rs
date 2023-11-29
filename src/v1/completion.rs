use serde::{Deserialize, Serialize};

#[derive(Serialize, Debug)]
pub struct Prompt(Vec<Modality>);

impl Prompt {
    pub fn empty() -> Self {
        Self(vec![])
    }

    /// Create a prompt from a single text item.
    pub fn from_text(text: impl Into<String>) -> Self {
        Self(vec![Modality::from_text(text)])
    }

    /// Create a multimodal prompt from a list of individual items with any modality.
    pub fn from_vec(items: Vec<Modality>) -> Self {
        Self(items)
    }
}

/// The prompt for models can be a combination of different modalities (Text and Image). The type of
/// modalities which are supported depend on the Model in question.
#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Modality {
    /// The only type of prompt which can be used with pure language models
    Text { data: String },
    /// An image input into the model. See [`Modality::from_image_path`].
    Image { data: String },
}

impl Modality {
    /// Instantiates a text prompt
    pub fn from_text(text: impl Into<String>) -> Self {
        Modality::Text { data: text.into() }
    }

    // pub fn from_image_path(path: impl AsRef<Path>) -> Result<Self, LoadImageError> {
    //     let bytes = image_preprocessing::from_image_path(path.as_ref())?;
    //     Ok(Self::from_image_bytes(&bytes))
    // }

    // /// Generates an image input from the binary representation of the image.
    // ///
    // /// Using this constructor you must use a binary representation compatible with the API. Png is
    // /// guaranteed to be supported, and all others formats are converted into it. Furthermore, the
    // /// model can only look at square shaped pictures. If the picture is not square shaped it will
    // /// be center cropped.
    // fn from_image_bytes(image: &[u8]) -> Self {
    //     Modality::Image {
    //         data: BASE64_STANDARD.encode(image).into(),
    //     }
    // }

    // /// Image input for model
    // ///
    // /// The model can only see squared pictures. Images are centercropped. You may want to use this
    // /// method instead of [`Self::from_image_path`] in case you have the image in memory already
    // /// and do not want to load it from a file again.
    // pub fn from_image(image: &DynamicImage) -> Result<Self, LoadImageError> {
    //     let bytes = image_preprocessing::preprocess_image(image);
    //     Ok(Self::from_image_bytes(&bytes))
    // }
}

#[derive(Serialize, Debug)]
pub struct CompletionRequest {
    /// Name of the model tasked with completing the prompt. E.g. `luminous-base"`.
    pub model: String,
    /// Prompt to complete. The modalities supported depend on `model`.
    pub prompt: Prompt,
    /// Limits the number of tokens, which are generated for the completion.
    pub maximum_tokens: u32,
    /// List of strings which will stop generation if they are generated. Stop sequences are
    /// helpful in structured texts. E.g.: In a question answering scenario a text may consist of
    /// lines starting with either "Question: " or "Answer: " (alternating). After producing an
    /// answer, the model will be likely to generate "Question: ". "Question: " may therefore be used
    /// as stop sequence in order not to have the model generate more questions but rather restrict
    /// text generation to the answers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_bias_inclusion: Option<Vec<String>>,
}

impl CompletionRequest {
    pub fn new(model: String, prompt: Prompt, maximum_tokens: u32) -> Self {
        Self {
            model,
            prompt,
            maximum_tokens,
            stop_sequences: None,
            temperature: None,
            top_k: None,
            top_p: None,
            completion_bias_inclusion: None,
        }
    }
}

#[derive(Deserialize, Debug)]
pub struct CompletionResponse {
    pub model_version: String,
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
