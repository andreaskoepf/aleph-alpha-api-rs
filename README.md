# aleph-alpha-api-rs

This is an unofficial Rust API client for the [Aleph Alpha API](https://app.aleph-alpha.com/). In contrast to Aleph Alpha's own [Rust client](https://github.com/Aleph-Alpha/aleph-alpha-client-rs) this implementation uses request and response structures which directly reflect the JSON requests and responses sent and received via HTTP. Data is owned by the request structures which avoids lifetime issues and makes it simple to clone and store requests.
The builder pattern can be used to fill the optional values of the request structs.

## Simple Text Completion Example

```rust
use aleph_alpha_api::{error::ApiError, Client, CompletionRequest, LUMINOUS_BASE};

const AA_API_TOKEN: &str = "<YOUR_AA_API_TOKEN>";

async fn print_completion() -> Result<(), ApiError> {
    let client = Client::new(AA_API_TOKEN.to_owned())?;

    let request =
        CompletionRequest::from_text(LUMINOUS_BASE.to_owned(), "An apple a day".to_owned(), 10)
            .temperature(0.8)
            .top_k(50)
            .top_p(0.95)
            .best_of(2)
            .minimum_tokens(2);

    let response = client.completion(&request, Some(true)).await?;

    println!("An apple a day{}", response.best_text());

    Ok(())
}

#[tokio::main]
async fn main() {
    print_completion().await.unwrap();
}
```