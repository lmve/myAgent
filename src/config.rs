//! 从环境变量（或 .env 文件）读取运行配置。

use anyhow::{bail, Context, Result};

use crate::context_manager::ContextStrategy;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_MODEL: &str = "gpt-4o-mini";
const DEFAULT_SYSTEM_PROMPT: &str = "你是一个乐于助人的 AI 助手，回答使用简体中文。";
const DEFAULT_TEMPERATURE: f32 = 0.7;
const DEFAULT_TIMEOUT_SECS: u64 = 120;
const DEFAULT_CONTEXT_MAX_TOKENS: usize = 6000;

/// API Key 的查找顺序，方便直接复用已有环境变量。
const API_KEY_VARS: [&str; 3] = ["LLM_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"];

#[derive(Debug, Clone)]
pub struct Config {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    /// 为 None 时表示不发送 system 消息。
    pub system_prompt: Option<String>,
    pub temperature: f32,
    pub timeout_secs: u64,
    pub stream: bool,
    pub context_strategy: ContextStrategy,
    pub context_max_tokens: usize,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let api_key = API_KEY_VARS
            .iter()
            .find_map(|key| std::env::var(key).ok().filter(|v| !v.trim().is_empty()))
            .with_context(|| {
                format!(
                    "没有找到 API key，请设置环境变量 {}（可参考 .env.example）",
                    API_KEY_VARS.join(" 或 ")
                )
            })?;

        let system_prompt =
            optional_env("LLM_SYSTEM_PROMPT").unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_string());
        let context_strategy = optional_env("CONTEXT_STRATEGY")
            .map(|value| ContextStrategy::parse(&value))
            .transpose()?
            .unwrap_or(ContextStrategy::Layered);
        let context_max_tokens = optional_env("CONTEXT_MAX_TOKENS")
            .map(|value| {
                value
                    .parse::<usize>()
                    .with_context(|| format!("CONTEXT_MAX_TOKENS 必须是正整数，收到 {value:?}"))
            })
            .transpose()?
            .unwrap_or(DEFAULT_CONTEXT_MAX_TOKENS);
        if context_max_tokens < 512 {
            bail!("CONTEXT_MAX_TOKENS 最小为 512");
        }

        Ok(Self {
            api_key,
            base_url: optional_env("LLM_BASE_URL").unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            model: optional_env("LLM_MODEL").unwrap_or_else(|| DEFAULT_MODEL.to_string()),
            // 允许通过空值显式关闭 system 消息
            system_prompt: (!system_prompt.trim().is_empty()).then_some(system_prompt),
            temperature: optional_env("LLM_TEMPERATURE")
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_TEMPERATURE),
            timeout_secs: optional_env("LLM_TIMEOUT_SECS")
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_TIMEOUT_SECS),
            stream: optional_env("LLM_STREAM")
                .map(|v| {
                    matches!(
                        v.trim().to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                })
                .unwrap_or(false),
            context_strategy,
            context_max_tokens,
        })
    }
}

fn optional_env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}
