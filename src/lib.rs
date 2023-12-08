//! # Inofficial Rust client library for the Aleph Alpha API
//! Example usage:
//! ```
//!use aleph_alpha_api::{error::ApiError, Client, CompletionRequest, LUMINOUS_BASE};
//!
//!const AA_API_TOKEN: &str = "<YOUR_AA_API_TOKEN>";
//!
//!async fn print_completion() -> Result<(), ApiError> {
//!    let client = Client::new(AA_API_TOKEN.to_owned())?;
//!
//!    let request =
//!        CompletionRequest::from_text(LUMINOUS_BASE.to_owned(), "An apple a day".to_owned(), 10)
//!            .temperature(0.8)
//!            .top_k(50)
//!            .top_p(0.95)
//!            .best_of(2)
//!            .minimum_tokens(2);
//!
//!    let response = client.completion(&request, Some(true)).await?;
//!
//!    println!("An apple a day{}", response.best_text());
//!
//!    Ok(())
//!}
//! ```

mod client;
mod completion;
mod embedding;
pub mod error;
mod evaluate;
mod explanation;
pub mod http;
pub mod image_processing;
mod tokenization;

pub const LUMINOUS_BASE: &str = "luminous-base";
pub const LUMINOUS_BASE_CONTROL: &str = "luminous-base-control";
pub const LUMINOUS_EXTENDED: &str = "luminous-extended";
pub const LUMINOUS_EXTENDED_CONTROL: &str = "luminous-extended-control";
pub const LUMINOUS_SUPREME: &str = "luminous-supreme";
pub const LUMINOUS_SUPREME_CONTROL: &str = "luminous-supreme-control";

pub use self::{
    client::Client, client::ALEPH_ALPHA_API_BASE_URL, completion::*, embedding::*, evaluate::*,
    explanation::*, tokenization::*,
};

// copied from https://github.com/dongri/openai-api-rs
#[macro_export]
macro_rules! impl_builder_methods {
    ($builder:ident, $($field:ident: $field_type:ty),*) => {
        impl $builder {
            $(
                pub fn $field(mut self, $field: $field_type) -> Self {
                    self.$field = Some($field);
                    self
                }
            )*
        }
    };
}
