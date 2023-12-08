use super::error::ApiError;
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::{header, Client, ClientBuilder, Error, StatusCode};

pub fn create_client(api_token: &str) -> Result<Client, Error> {
    let mut headers = HeaderMap::new();

    let mut auth_value = HeaderValue::from_str(&format!("Bearer {api_token}")).unwrap();
    // Consider marking security-sensitive headers with `set_sensitive`.
    auth_value.set_sensitive(true);
    headers.insert(header::AUTHORIZATION, auth_value);

    Ok(ClientBuilder::new().default_headers(headers).build()?)
}

pub async fn translate_http_error(
    response: reqwest::Response,
) -> Result<reqwest::Response, ApiError> {
    let status = response.status();
    if !status.is_success() {
        // Store body in a variable, so we can use it, even if it is not an Error emitted by
        // the API, but an intermediate Proxy like NGinx, so we can still forward the error
        // message.
        let body = response.text().await?;
        let translated_error = match status {
            StatusCode::TOO_MANY_REQUESTS => ApiError::TooManyRequests,
            StatusCode::SERVICE_UNAVAILABLE => ApiError::Busy,
            _ => ApiError::Http {
                status: status.as_u16(),
                body,
            },
        };
        Err(translated_error)
    } else {
        Ok(response)
    }
}

pub async fn get(
    client: &reqwest::Client,
    base_url: &str,
    path: &str,
    query: Option<Vec<(String, String)>>,
) -> Result<reqwest::Response, ApiError> {
    let url = format!("{base_url}{path}");
    let mut request = client.get(url);
    println!("{:?}", request);
    if let Some(q) = query {
        request = request.query(&q);
    }
    let response = request.send().await?;
    println!("response: {:?}", response);
    translate_http_error(response).await
}

pub async fn delete(
    client: &reqwest::Client,
    base_url: &str,
    path: &str,
) -> Result<reqwest::Response, ApiError> {
    let url = format!("{base_url}{path}");
    let response = client.delete(url).send().await?;
    translate_http_error(response).await
}
