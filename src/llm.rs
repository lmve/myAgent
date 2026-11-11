//! 一个最小的 OpenAI 兼容 Chat Completions 客户端。
//!
//! 之所以直接说“OpenAI 兼容”，是因为 OpenAI / DeepSeek / 通义 / Moonshot /
//! Ollama 等绝大多数服务都实现了同样的 `POST {base_url}/chat/completions` 协议，
//! 只要换 `LLM_BASE_URL` 和 `LLM_MODEL` 即可切换。

use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::Config;

/// 对话消息，请求和响应共用。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    pub fn assistant(content: Option<String>, tool_calls: Option<Vec<ToolCall>>) -> Self {
        Self {
            role: "assistant".into(),
            content,
            tool_calls,
            tool_call_id: None,
        }
    }

    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    tools: &'a [Value],
    stream: bool,
    temperature: f32,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    #[serde(default)]
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    #[serde(default)]
    message: Option<Message>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Clone, Copy, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
}

/// 一次调用的完整结果。
#[derive(Debug)]
pub struct ChatOutput {
    pub content: String,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub finish_reason: Option<String>,
    pub usage: Option<Usage>,
}

pub struct LlmClient {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
    temperature: f32,
}

impl LlmClient {
    pub fn new(config: &Config) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .context("构造 HTTP 客户端失败")?;

        Ok(Self {
            http,
            base_url: config.base_url.trim_end_matches('/').to_string(),
            api_key: config.api_key.clone(),
            model: config.model.clone(),
            temperature: config.temperature,
        })
    }

    /// 发送整段对话历史和可用工具，返回模型回复或工具调用。
    pub async fn chat(&self, messages: &[Message], tools: &[Value]) -> Result<ChatOutput> {
        let url = format!("{}/chat/completions", self.base_url);
        let body = ChatRequest {
            model: &self.model,
            messages,
            tools,
            stream: false,
            temperature: self.temperature,
        };

        let response = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("请求 {url} 失败（检查网络、LLM_BASE_URL 是否正确）"))?;

        let status = response.status();
        let raw = response.text().await.context("读取响应体失败")?;

        if !status.is_success() {
            bail!("LLM 接口返回 HTTP {status}：{}", preview(&raw));
        }

        let parsed: ChatResponse = serde_json::from_str(&raw)
            .with_context(|| format!("解析响应失败：{}", preview(&raw)))?;
        extract_output(parsed)
    }
}

fn extract_output(response: ChatResponse) -> Result<ChatOutput> {
    let choice = response
        .choices
        .into_iter()
        .next()
        .context("LLM 响应中没有 choices")?;

    let message = choice.message.context("LLM 响应的 choice 中没有 message")?;
    let content = message.content.unwrap_or_default().trim().to_string();
    let tool_calls = message.tool_calls.filter(|calls| !calls.is_empty());

    if content.is_empty() && tool_calls.is_none() {
        bail!(
            "LLM 返回了空内容（finish_reason = {}）",
            choice.finish_reason.as_deref().unwrap_or("unknown")
        );
    }

    Ok(ChatOutput {
        content,
        tool_calls,
        finish_reason: choice.finish_reason,
        usage: response.usage,
    })
}

/// 截断过长的响应体，避免错误信息刷屏。
fn preview(text: &str) -> String {
    const MAX: usize = 800;
    if text.chars().count() <= MAX {
        text.to_string()
    } else {
        let head: String = text.chars().take(MAX).collect();
        format!("{head}…(已截断)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_normal_response() {
        let raw = r#"{
            "choices": [
                {"message": {"role": "assistant", "content": " 你好 "}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
        }"#;
        let out = extract_output(serde_json::from_str(raw).unwrap()).unwrap();
        assert_eq!(out.content, "你好");
        assert!(out.tool_calls.is_none());
        assert_eq!(out.finish_reason.as_deref(), Some("stop"));
        assert_eq!(out.usage.unwrap().total_tokens, 13);
    }

    #[test]
    fn rejects_empty_choices() {
        let raw = r#"{"choices": []}"#;
        let err = extract_output(serde_json::from_str(raw).unwrap()).unwrap_err();
        assert!(err.to_string().contains("choices"));
    }

    #[test]
    fn rejects_empty_content() {
        let raw = r#"{"choices": [{"message": {"role": "assistant", "content": "  "}, "finish_reason": "length"}]}"#;
        let err = extract_output(serde_json::from_str(raw).unwrap()).unwrap_err();
        assert!(err.to_string().contains("length"));
    }

    #[test]
    fn parses_tool_call_without_content() {
        let raw = r#"{"choices": [{"message": {
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "add", "arguments": "{\"a\":1,\"b\":2}"}
            }]
        }, "finish_reason": "tool_calls"}]}"#;
        let out = extract_output(serde_json::from_str(raw).unwrap()).unwrap();
        assert_eq!(out.content, "");
        assert_eq!(out.tool_calls.unwrap()[0].function.name, "add");
    }

    #[test]
    fn preview_truncates_long_text() {
        let long = "a".repeat(1000);
        assert!(preview(&long).ends_with("已截断)"));
    }
}
