//! 一个最小的 OpenAI 兼容 Chat Completions 客户端。
//!
//! 之所以直接说“OpenAI 兼容”，是因为 OpenAI / DeepSeek / 通义 / Moonshot /
//! Ollama 等绝大多数服务都实现了同样的 `POST {base_url}/chat/completions` 协议，
//! 只要换 `LLM_BASE_URL` 和 `LLM_MODEL` 即可切换。

use std::time::Duration;
use std::{collections::BTreeMap, str};

use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
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
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: StreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct StreamDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ToolCallDelta>,
}

#[derive(Debug, Deserialize)]
struct ToolCallDelta {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(rename = "type", default)]
    call_type: Option<String>,
    #[serde(default)]
    function: Option<FunctionCallDelta>,
}

#[derive(Debug, Default, Deserialize)]
struct FunctionCallDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
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

    /// 根据 stream 选择普通 JSON 响应或 SSE 流式响应。
    pub async fn chat<F>(
        &self,
        messages: &[Message],
        tools: &[Value],
        stream: bool,
        on_delta: F,
    ) -> Result<ChatOutput>
    where
        F: FnMut(&str),
    {
        if stream {
            self.chat_stream(messages, tools, on_delta).await
        } else {
            self.chat_non_stream(messages, tools).await
        }
    }

    async fn chat_non_stream(&self, messages: &[Message], tools: &[Value]) -> Result<ChatOutput> {
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

    async fn chat_stream<F>(
        &self,
        messages: &[Message],
        tools: &[Value],
        mut on_delta: F,
    ) -> Result<ChatOutput>
    where
        F: FnMut(&str),
    {
        let url = format!("{}/chat/completions", self.base_url);
        let body = ChatRequest {
            model: &self.model,
            messages,
            tools,
            stream: true,
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
        if !status.is_success() {
            let raw = response.text().await.context("读取错误响应体失败")?;
            bail!("LLM 接口返回 HTTP {status}：{}", preview(&raw));
        }

        let mut response_stream = response.bytes_stream();
        let mut pending = Vec::new();
        let mut content = String::new();
        let mut tool_calls = BTreeMap::new();
        let mut finish_reason = None;
        let mut usage = None;
        let mut done = false;

        while let Some(chunk) = response_stream.next().await {
            let chunk = chunk.context("读取 LLM 流式响应失败")?;
            pending.extend_from_slice(&chunk);

            while let Some(newline) = pending.iter().position(|byte| *byte == b'\n') {
                let mut line: Vec<u8> = pending.drain(..=newline).collect();
                line.pop();
                process_sse_line(
                    &line,
                    &mut content,
                    &mut tool_calls,
                    &mut finish_reason,
                    &mut usage,
                    &mut on_delta,
                    &mut done,
                )?;
                if done {
                    break;
                }
            }
            if done {
                break;
            }
        }

        if !done && !pending.is_empty() {
            process_sse_line(
                &pending,
                &mut content,
                &mut tool_calls,
                &mut finish_reason,
                &mut usage,
                &mut on_delta,
                &mut done,
            )?;
        }

        let tool_calls: Vec<_> = tool_calls.into_values().collect();
        if content.trim().is_empty() && tool_calls.is_empty() {
            bail!("LLM 流式响应没有内容或工具调用");
        }

        Ok(ChatOutput {
            content: content.trim().to_string(),
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            finish_reason,
            usage,
        })
    }
}

fn process_sse_line<F>(
    line: &[u8],
    content: &mut String,
    tool_calls: &mut BTreeMap<usize, ToolCall>,
    finish_reason: &mut Option<String>,
    usage: &mut Option<Usage>,
    on_delta: &mut F,
    done: &mut bool,
) -> Result<()>
where
    F: FnMut(&str),
{
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    let Ok(line) = str::from_utf8(line) else {
        bail!("LLM 流式响应包含无效 UTF-8");
    };
    let Some(data) = line.strip_prefix("data:") else {
        return Ok(());
    };
    let data = data.trim_start();
    if data == "[DONE]" {
        *done = true;
        return Ok(());
    }
    if data.is_empty() {
        return Ok(());
    }

    let chunk: StreamChunk = serde_json::from_str(data).context("解析 LLM 流式事件失败")?;
    if let Some(chunk_usage) = chunk.usage {
        *usage = Some(chunk_usage);
    }
    for choice in chunk.choices {
        if choice.finish_reason.is_some() {
            *finish_reason = choice.finish_reason;
        }
        if let Some(delta) = choice.delta.content {
            content.push_str(&delta);
            on_delta(&delta);
        }
        for delta in choice.delta.tool_calls {
            let call = tool_calls.entry(delta.index).or_insert_with(|| ToolCall {
                id: String::new(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: String::new(),
                    arguments: String::new(),
                },
            });
            if let Some(id) = delta.id {
                call.id.push_str(&id);
            }
            if let Some(call_type) = delta.call_type {
                call.call_type = call_type;
            }
            if let Some(function) = delta.function {
                if let Some(name) = function.name {
                    call.function.name.push_str(&name);
                }
                if let Some(arguments) = function.arguments {
                    call.function.arguments.push_str(&arguments);
                }
            }
        }
    }
    Ok(())
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
    fn processes_streamed_content_and_tool_call_fragments() {
        let mut content = String::new();
        let mut tool_calls = BTreeMap::new();
        let mut finish_reason = None;
        let mut usage = None;
        let mut printed = String::new();
        let mut done = false;
        let mut on_delta = |delta: &str| printed.push_str(delta);

        process_sse_line(
            br#"data: {"choices":[{"delta":{"content":"Hello "},"finish_reason":null}]}"#,
            &mut content,
            &mut tool_calls,
            &mut finish_reason,
            &mut usage,
            &mut on_delta,
            &mut done,
        )
        .unwrap();
        process_sse_line(
            br#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"ad","arguments":"{\"a\":"}}]},"finish_reason":null}]}"#,
            &mut content,
            &mut tool_calls,
            &mut finish_reason,
            &mut usage,
            &mut on_delta,
            &mut done,
        )
        .unwrap();
        process_sse_line(
            br#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"d","arguments":"1,\"b\":2}"}}]},"finish_reason":"tool_calls"}]}"#,
            &mut content,
            &mut tool_calls,
            &mut finish_reason,
            &mut usage,
            &mut on_delta,
            &mut done,
        )
        .unwrap();

        assert_eq!(content, "Hello ");
        assert_eq!(printed, "Hello ");
        assert_eq!(finish_reason.as_deref(), Some("tool_calls"));
        let call = &tool_calls[&0];
        assert_eq!(call.id, "call_1");
        assert_eq!(call.function.name, "add");
        assert_eq!(call.function.arguments, r#"{"a":1,"b":2}"#);
    }

    #[test]
    fn recognizes_stream_done_event() {
        let mut content = String::new();
        let mut tool_calls = BTreeMap::new();
        let mut finish_reason = None;
        let mut usage = None;
        let mut done = false;

        process_sse_line(
            b"data: [DONE]",
            &mut content,
            &mut tool_calls,
            &mut finish_reason,
            &mut usage,
            &mut |_| {},
            &mut done,
        )
        .unwrap();

        assert!(done);
    }

    #[test]
    fn preview_truncates_long_text() {
        let long = "a".repeat(1000);
        assert!(preview(&long).ends_with("已截断)"));
    }
}
