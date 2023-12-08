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

## Running the Sampling Report Example

The sampling report example generates completions of 250 random prompts that were collected as part of the [Open-Assistant](https://github.com/LAION-AI/Open-Assistant/) project.

The sampling parameters used for the completions are specified via a JSON configuration file. You find two example configurations for general models and instruction tuned models (the *-control variants): 
- [sampling_default.json](examples/config/sampling_default.json)
- [sampling_control.json](examples/config/sampling_control.json)

Specify the configuration file to use via the `--config` command line argument.

Please make sure to set your API token as `AA_API_TOKEN` environment variable before launching the example:

```bash
export AA_API_TOKEN=<YOUR_AA_API_TOKEN>
cargo run --example sampling_report -- --config examples/config/sampling_default.json --model luminous-base
```