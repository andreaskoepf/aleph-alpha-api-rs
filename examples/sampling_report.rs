use chrono::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fmt::Write;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use tokio;

use aleph_alpha_api::v1::client::Client;
use aleph_alpha_api::v1::completion::CompletionRequest;
use aleph_alpha_api::v1::completion::Prompt;
use clap::Parser;
use json;
use serde::Serialize;

fn read_prompts_from_jsonl(
    input_file_name: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let file = File::open(input_file_name)?;
    let mut buf_reader = BufReader::new(file);
    let mut prompts: Vec<String> = Vec::new();
    loop {
        let mut line = String::new();
        let len = buf_reader.read_line(&mut line)?;
        if len == 0 {
            break;
        }

        if let Some(prompt) = json::parse(&line)?.as_str() {
            prompts.push(prompt.to_owned());
        }
    }
    Ok(prompts)
}

#[derive(Serialize, Deserialize, Clone, Default, Debug)]
struct GenerationArgs {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_new_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_new_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_optimizations: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_of: Option<u32>,
}

#[derive(Deserialize, Clone, Default, Debug)]
struct SamplingConfiguration {
    pub generate_args: GenerationArgs,
    pub system_prompt: Option<String>,
    pub assistant_name: Option<String>,
    pub user_name: Option<String>,
}

type NamedConfigurations = HashMap<String, SamplingConfiguration>;

#[derive(Parser, Debug, Serialize, Clone)]
struct Args {
    /// File name of jsonl file containing prompts
    #[arg(long, default_value = "examples/prompts_oa_en.jsonl")]
    prompts: String,

    #[arg(long, required = true)]
    config: String,

    #[arg(long, default_value_t = true)]
    nice: bool,

    #[arg(long, default_value = "luminous-base")]
    model: String,

    #[arg(long)]
    report: Option<String>,
}

#[derive(Serialize, Debug)]
struct PromptResult {
    pub sampling_config: String,
    pub sampling_params: GenerationArgs,
    pub outputs: Vec<String>,
}

#[derive(Serialize, Debug)]
struct SamplingResult {
    pub prompt: String,
    pub results: Vec<PromptResult>,
}

impl SamplingResult {
    fn new(prompt: String) -> Self {
        Self {
            prompt,
            results: Vec::new(),
        }
    }
}

#[derive(Serialize, Debug)]
struct SamplingReport {
    pub model_name: String,
    pub date: String,
    pub args: Args,
    pub prompts: Vec<SamplingResult>,
}

fn read_configuration(file_name: &str) -> NamedConfigurations {
    let mut file = File::open(file_name).expect("Could not open configuration file.");
    let mut contents: String = String::new();
    file.read_to_string(&mut contents)
        .expect("reading configuration file failed.");

    let config: NamedConfigurations = serde_json::from_str(contents.as_str()).unwrap();
    config
}

fn merge_with_default(
    source: &SamplingConfiguration,
    default_config: Option<&SamplingConfiguration>,
) -> SamplingConfiguration {
    let mut cfg: SamplingConfiguration = if let Some(defaults) = default_config {
        (*defaults).clone()
    } else {
        SamplingConfiguration::default()
    };

    if let Some(max_new_tokens) = source.generate_args.max_new_tokens {
        cfg.generate_args.max_new_tokens = Some(max_new_tokens);
    }
    if let Some(min_new_tokens) = source.generate_args.min_new_tokens {
        cfg.generate_args.min_new_tokens = Some(min_new_tokens);
    }
    if let Some(temperature) = source.generate_args.temperature {
        cfg.generate_args.temperature = Some(temperature);
    }
    if let Some(presence_penalty) = source.generate_args.presence_penalty {
        cfg.generate_args.presence_penalty = Some(presence_penalty);
    }
    if let Some(frequency_penalty) = source.generate_args.frequency_penalty {
        cfg.generate_args.frequency_penalty = Some(frequency_penalty);
    }
    if let Some(disable_optimizations) = source.generate_args.disable_optimizations {
        cfg.generate_args.disable_optimizations = Some(disable_optimizations);
    }
    if let Some(best_of) = source.generate_args.best_of {
        cfg.generate_args.best_of = Some(best_of);
    }
    if let Some(n) = source.generate_args.n {
        cfg.generate_args.n = Some(n);
    }

    if let Some(system_prompt) = &source.system_prompt {
        cfg.system_prompt = Some(system_prompt.clone());
    }
    if let Some(assistant_name) = &source.assistant_name {
        cfg.assistant_name = Some(assistant_name.clone());
    }
    if let Some(user_name) = &source.user_name {
        cfg.user_name = Some(user_name.clone());
    }

    cfg
}

fn format_prompt(prompt: &str, sampling_config: &SamplingConfiguration) -> String {
    let user_name = sampling_config
        .user_name
        .as_ref()
        .expect("user name must be specified");
    let assistant_name = sampling_config
        .assistant_name
        .as_ref()
        .expect("assistant name must be specified");

    let mut input_text: String = String::new();

    if let Some(system_prompt) = sampling_config.system_prompt.as_ref() {
        write!(input_text, "{}\n", system_prompt).unwrap();
    }

    write!(input_text, "{user_name} {prompt}\n{assistant_name}").unwrap();

    input_text
}

fn configure_request(req: &mut CompletionRequest, args: &GenerationArgs) {
    if let Some(t) = args.temperature {
        req.temperature = Some(t);
    }
    if let Some(min_tokens) = args.min_new_tokens {
        req.minimum_tokens = Some(min_tokens);
    }
    if let Some(max_tokens) = args.max_new_tokens {
        req.maximum_tokens = max_tokens;
    }
    if let Some(top_k) = args.top_k {
        req.top_k = Some(top_k);
    }
    if let Some(top_p) = args.top_p {
        req.top_p = Some(top_p);
    }
    if let Some(presence_penalty) = args.presence_penalty {
        req.presence_penalty = Some(presence_penalty);
        req.repetition_penalties_include_completion = Some(true);
        //req.use_multiplicative_presence_penalty = Some(true);
    }
    if let Some(frequency_penalty) = args.frequency_penalty {
        req.frequency_penalty = Some(frequency_penalty);
        req.repetition_penalties_include_completion = Some(true);
    }
    if let Some(disable_optimizations) = args.disable_optimizations {
        req.disable_optimizations = Some(disable_optimizations);
    }
    if let Some(best_of) = args.best_of {
        req.best_of = Some(best_of);
    }
    if let Some(n) = args.n {
        req.n = Some(n);
    }
}

async fn sample_all(
    client: &Client,
    configurations: &NamedConfigurations,
    prompt: &str,
    args: &Args,
) -> SamplingResult {
    let default_config = configurations.get("default");
    let mut result = SamplingResult::new(prompt.to_owned());
    for (name, configuration) in configurations.into_iter() {
        if name == "default" {
            continue;
        }

        let configuration = merge_with_default(configuration, default_config);

        let formatted_prompt = format_prompt(prompt, &configuration);

        let model = &args.model;
        let nice = args.nice;

        let mut req = CompletionRequest::new(
            model.to_owned(),
            Prompt::from_text(formatted_prompt.clone()),
            100,
        );

        req.stop_sequences = Some(vec![configuration.user_name.unwrap().clone()]);
        configure_request(&mut req, &configuration.generate_args);

        let response = client.completion(&req, Some(nice)).await.unwrap();
        println!("{}", response.best_text());

        let prompt_result = PromptResult {
            sampling_config: name.clone(),
            sampling_params: configuration.generate_args.clone(),
            outputs: response
                .completions
                .iter()
                .map(|x| x.completion.clone())
                .collect(),
        };
        result.results.push(prompt_result);
    }
    result
}

fn write_report_file(file_name: &str, report: &SamplingReport) {
    let file = File::create(file_name).expect("Could not create report file.");
    serde_json::to_writer_pretty(file, &report).unwrap();
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let api_token = std::env::var("AA_API_TOKEN")
        .expect("AA_API_TOKEN environment variable must be specified to run sample.");

    let config = read_configuration(&args.config);
    println!("{:?}", config);
    let prompts = read_prompts_from_jsonl(&args.prompts).unwrap();

    let client = Client::new(api_token).expect("Could not create API client");

    let mut report: SamplingReport = SamplingReport {
        model_name: args.model.clone(),
        date: Utc::now().to_rfc3339(),
        args: args.clone(),
        prompts: Vec::new(),
    };

    let report_file_name = if let Some(file_name) = &args.report {
        file_name.clone()
    } else {
        let safe_model_name = report
            .model_name
            .chars()
            .map(|c| {
                if c.is_ascii() && (c.is_alphanumeric() || c == '-') {
                    c
                } else {
                    '_'
                }
            })
            .collect::<String>();
        let date = report.date.split('T').take(1).nth(0).unwrap();
        let config_name = Path::new(&args.config)
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap();
        format!("{}_{}_{}.json", date, safe_model_name, config_name)
    };

    for prompt in prompts {
        let prompt_result = sample_all(&client, &config, &prompt, &args).await;
        report.prompts.push(prompt_result);
        write_report_file(&report_file_name, &report);
    }
}
