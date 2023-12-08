use aleph_alpha_api::{Client, CompletionRequest, LUMINOUS_BASE};
use clap::Parser;
use tokio;

#[derive(Parser, Debug)]
#[command(about = "Generate a prompt completion via the Aleph Alpha inference API")]
struct Args {
    /// Model name
    #[arg(long, default_value = LUMINOUS_BASE)]
    model: String,

    /// Maximum number of tokens to generate
    #[arg(long, default_value_t = 64)]
    max_tokens: u32,

    /// The API token to use (otherwise use AA_API_TOKEN environment variable)
    #[arg(long)]
    api_token: Option<String>,

    #[arg(long)]
    top_p: Option<f64>,

    #[arg(long)]
    top_k: Option<u32>,

    /// The text used as pre-text for the generation.
    prompt: String,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let api_token = args.api_token.unwrap_or_else(|| {
        std::env::var("AA_API_TOKEN")
            .expect("AA_API_TOKEN environment variable must be specified to run sample.")
    });

    let client = Client::new(api_token).expect("Failed to create API client");
    let mut req = CompletionRequest::from_text(args.model, args.prompt, args.max_tokens);

    req.top_k = args.top_k;
    req.top_p = args.top_p;

    println!("Sending request: {:#?}", req);

    let res = client.completion(&req, Some(true)).await.unwrap();

    for c in res.completions {
        println!(
            "Completion: \"{}\"\nFinish reason: {}",
            c.completion, c.finish_reason
        )
    }
}
