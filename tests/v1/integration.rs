use aleph_alpha_api_rs::v1::client::{ApiError, Client};
use aleph_alpha_api_rs::v1::completion::{CompletionRequest, Prompt};
use dotenv::dotenv;
use lazy_static::lazy_static;

lazy_static! {
    static ref AA_API_TOKEN: String = {
        // Use `.env` file if it exists
        let _ = dotenv();
        std::env::var("AA_API_TOKEN")
            .expect("AA_API_TOKEN environment variable must be specified to run tests.")
    };
}

#[tokio::test]
async fn completion_with_luminous_base() {
    // When

    let client = Client::new(AA_API_TOKEN.clone()).expect("failed to create client");
    let req = CompletionRequest::new(
        "luminous-base".into(),
        Prompt::from_text("Hallo wie geht es dir? "),
        20,
    );
    let response = client.completion(req).await.unwrap();

    assert!(!response.completions.is_empty());
    assert!(!response.best_text().is_empty());
    println!("{:?}", response);
}
