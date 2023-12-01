use aleph_alpha_api_rs::v1::api_tokens::{CreateApiTokenRequest, CreateApiTokenResponse};
use aleph_alpha_api_rs::v1::client::Client;
use aleph_alpha_api_rs::v1::completion::{CompletionRequest, Prompt};
use aleph_alpha_api_rs::v1::evaluate::EvaluationRequest;
use aleph_alpha_api_rs::v1::tokenization::{DetokenizationRequest, TokenizationRequest};
use aleph_alpha_api_rs::v1::users::UserDetail;

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
    let response = client.completion(&req, Some(true)).await.unwrap();

    assert!(!response.completions.is_empty());
    assert!(!response.best_text().is_empty());
    println!("{:?}", response);
}

#[tokio::test]
async fn completion_with_luminous_base_token_ids() {
    // Given
    let client = Client::new(AA_API_TOKEN.clone()).expect("failed to create client");
    let prompt = Prompt::from_token_ids(vec![49222, 15, 5390, 4], None);

    // When
    let mut req = CompletionRequest::new("luminous-base".into(), prompt, 20);
    req.echo = Some(true);

    let response = client.completion(&req, Some(true)).await.unwrap();

    // Then
    assert!(!response.completions.is_empty());
    assert!(!response.best_text().is_empty());
    assert!(response.best_text().contains("Hello, World!"));
    println!("{:?}", response);
}

#[tokio::test]
async fn evaluate_with_luminous_base() {
    let model = "luminous-base";
    let prompt = Prompt::from_text("An apple a day keeps the");
    let completion_expected = " doctor away";
    let client = Client::new(AA_API_TOKEN.clone()).expect("failed to create client");

    let req = EvaluationRequest {
        model: model.to_owned(),
        prompt,
        hosting: None,
        completion_expected: completion_expected.to_owned(),
        contextual_control_threshold: None,
        control_log_additive: None,
    };

    let response = client.evaluate(&req, Some(true)).await.unwrap();
    assert!(!response.model_version.is_empty());
    assert_eq!(response.result.correct_greedy, Some(true));
    println!("{:?}", response)
}

#[tokio::test]
async fn tokenization_with_luminous_base() {
    // Given
    let model = "luminous-base";
    let input = "Hello, World!";
    let client = Client::new(AA_API_TOKEN.clone()).unwrap();

    // When
    let request1 = TokenizationRequest {
        model: model.to_owned(),
        prompt: input.to_owned(),
        tokens: false,
        token_ids: true,
    };
    let response1 = client.tokenize(&request1).await.unwrap();

    let request2 = TokenizationRequest {
        token_ids: false,
        tokens: true,
        ..request1
    };
    let response2 = client.tokenize(&request2).await.unwrap();

    // Then
    assert_eq!(response1.tokens, None);
    assert_eq!(response1.token_ids, Some(vec![49222, 15, 5390, 4]));

    assert_eq!(response2.token_ids, None);
    assert_eq!(
        response2.tokens,
        Some(
            vec!["ĠHello", ",", "ĠWorld", "!"]
                .into_iter()
                .map(str::to_owned)
                .collect()
        )
    );
}

#[tokio::test]
async fn detokenization_with_luminous_base() {
    // Given
    let model = "luminous-base";
    let input = vec![49222, 15, 5390, 4];
    let client = Client::new(AA_API_TOKEN.clone()).unwrap();

    // When
    let task = DetokenizationRequest {
        model: model.to_owned(),
        token_ids: input.clone(),
    };

    let response = client.detokenize(&task).await.unwrap();

    // Then
    assert!(response.result.contains("Hello, World!"));
}

#[tokio::test]
async fn list_api_tokens() {
    // Given
    let client = Client::new(AA_API_TOKEN.clone()).unwrap();

    // When
    let api_tokens = client.list_api_tokens().await.unwrap();

    println!("{:?}", api_tokens);
    assert!(!api_tokens.is_empty());
}

#[tokio::test]
#[ignore]
async fn create_and_delete_api_token() {
    let client = Client::new(AA_API_TOKEN.clone()).unwrap();

    let create_req = CreateApiTokenRequest {
        description: "A test token".to_string(),
    };
    let create_res: CreateApiTokenResponse = client.create_api_token(&create_req).await.unwrap();
    assert!(!create_res.token.is_empty());

    client
        .delete_api_token(create_res.metadata.token_id)
        .await
        .unwrap();

    println!("{:?}", create_res);
}

#[tokio::test]
async fn get_user_settings() {
    // Given
    let client = Client::new(AA_API_TOKEN.clone()).unwrap();

    // When
    let user_detail: UserDetail = client.get_user_settings().await.unwrap();

    println!("{:?}", user_detail);

    assert!(user_detail.email.contains("@"))
}
