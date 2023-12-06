pub mod api_tokens;
pub mod client;
pub mod completion;
pub mod embedding;
pub mod error;
pub mod evaluate;
pub mod explanation;
pub mod http;
pub mod tokenization;
pub mod users;

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
