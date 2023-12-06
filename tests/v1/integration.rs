use aleph_alpha_api_rs::v1::api_tokens::{CreateApiTokenRequest, CreateApiTokenResponse};
use aleph_alpha_api_rs::v1::client::Client;
use aleph_alpha_api_rs::v1::completion::{CompletionRequest, Prompt};
use aleph_alpha_api_rs::v1::embedding::{
    EmbeddingRepresentation, EmbeddingRequest, SemanticEmbeddingRequest,
};
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
    let prompt = "An apple a day keeps the";
    let completion_expected = " doctor away";
    let client = Client::new(AA_API_TOKEN.clone()).expect("failed to create client");

    let req = EvaluationRequest::from_text(model, prompt, completion_expected);

    let response = client.evaluate(&req, Some(true)).await.unwrap();
    println!("{:?}", response);

    assert!(!response.model_version.is_empty());
    assert_eq!(response.result.correct_greedy, Some(true));
}

#[tokio::test]
async fn evaluate_with_luminous_base_flat_earth() {
    let model = "luminous-base";
    let prompt = "The earth is flat. This statement is";
    let completion_false = " false.";
    let completion_true = " true.";
    let client = Client::new(AA_API_TOKEN.clone()).expect("failed to create client");

    let req_false = EvaluationRequest::from_text(model, prompt, completion_false);
    let req_true = EvaluationRequest::from_text(model, prompt, completion_true);

    let response_false = client.evaluate(&req_false, Some(true)).await.unwrap();
    let response_true = client.evaluate(&req_true, Some(true)).await.unwrap();

    assert!(!response_false.model_version.is_empty());

    println!("response_false: {:?}", response_false);
    println!("response_true: {:?}", response_true);

    assert!(
        response_false.result.log_perplexity_per_token.unwrap()
            < response_true.result.log_perplexity_per_token.unwrap()
    );
}

#[tokio::test]
async fn embed_with_luminous_base() {
    let client = Client::new(AA_API_TOKEN.clone()).expect("failed to create client");

    let model = "luminous-base";
    let text_prompt = "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua.";
    let req = EmbeddingRequest::from_text(model, text_prompt, 1, "max", true);

    let response = client.embed(&req, Some(true)).await.unwrap();

    assert_eq!(response.embeddings.len(), 1);
    assert!(response.embeddings.get("layer_1").is_some());
    assert!(response.embeddings["layer_1"].get("max").is_some());
    assert!(response.embeddings["layer_1"]["max"].len() > 64);
}

#[tokio::test]
async fn semantic_embed_with_luminous_base() {
    let client = Client::new(AA_API_TOKEN.clone()).expect("failed to create client");
    let model = "luminous-base";
    let prompt = Prompt::from_text("An apple a day keeps the doctor away.");

    let req = SemanticEmbeddingRequest {
        model: model.to_owned(),
        prompt: prompt,
        representation: EmbeddingRepresentation::Symmetric,
        compress_to_size: Some(128),
        ..Default::default()
    };

    let response = client.semantic_embed(&req, Some(true)).await.unwrap();
    assert_eq!(response.embedding.len(), 128);
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
async fn download_tokenizer_luminous_base() {
    // Given
    let model = "luminous-base";
    let client = Client::new(AA_API_TOKEN.clone()).unwrap();
    let input: &str = "This is a test";

    // When
    let tokenizer = client.get_tokenizer(model).await.unwrap();
    let encoding = tokenizer.encode(input, false).unwrap();

    // Then
    assert!(encoding.get_ids().len() > 0);
    assert_eq!(encoding.get_ids(), [1730, 387, 247, 3173]);
}

#[tokio::test]
async fn tokenizer_cross_check_luminous_base() {
    // Given
    let model = "luminous-base";
    let client = Client::new(AA_API_TOKEN.clone()).unwrap();
    let input: &str = "the cat is on the mat";

    // When
    let tokenizer = client.get_tokenizer(model).await.unwrap();
    let encoding = tokenizer.encode(input, false).unwrap();
    let tokenization_response = client
        .tokenize(&TokenizationRequest {
            model: model.to_owned(),
            prompt: input.to_owned(),
            tokens: true,
            token_ids: true,
        })
        .await
        .unwrap();

    // Then
    assert_eq!(encoding.get_ids(), tokenization_response.token_ids.unwrap());
    assert_eq!(encoding.get_tokens(), tokenization_response.tokens.unwrap());
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
