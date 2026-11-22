use anyhow::{bail, Context, Result};

use crate::llm::{LlmClient, Message};

const SUMMARY_PREFIX: &str = "[Agent Context Summary]\n";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextStrategy {
    Truncate,
    Summarize,
    Layered,
}

fn log_compaction(strategy: ContextStrategy, before: usize, after: usize, dropped_turns: usize) {
    eprintln!(
        "[context] strategy={} estimated_units={} -> {}, dropped_turns={dropped_turns}",
        strategy.as_str(),
        before,
        after
    );
}

impl ContextStrategy {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "truncate" => Ok(Self::Truncate),
            "summarize" => Ok(Self::Summarize),
            "layered" => Ok(Self::Layered),
            _ => bail!("无效的 CONTEXT_STRATEGY={value:?}，可选值为 truncate、summarize、layered"),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Truncate => "truncate",
            Self::Summarize => "summarize",
            Self::Layered => "layered",
        }
    }
}

pub struct ContextManager {
    strategy: ContextStrategy,
    max_tokens: usize,
}

impl ContextManager {
    pub fn new(strategy: ContextStrategy, max_tokens: usize) -> Self {
        Self {
            strategy,
            max_tokens,
        }
    }

    /// Compresses only when needed and keeps assistant tool-call/result groups intact.
    pub async fn prepare(&self, history: &mut Vec<Message>, client: &LlmClient) -> Result<()> {
        let system_messages: Vec<_> = history
            .iter()
            .filter(|message| message.role == "system" && !is_summary(message))
            .cloned()
            .collect();
        let previous_summary = history
            .iter()
            .find(|message| is_summary(message))
            .and_then(|message| message.content.as_deref())
            .map(|content| content[SUMMARY_PREFIX.len()..].to_string())
            .unwrap_or_default();
        let turns = split_turns(history);
        let original_estimate = estimate_messages(&system_messages)
            + estimate_summary(&previous_summary)
            + estimate_turns(&turns);

        if original_estimate <= self.max_tokens {
            return Ok(());
        }

        let system_cost = estimate_messages(&system_messages);
        if system_cost >= self.max_tokens {
            bail!(
                "system prompt 估算长度已达到上下文上限 {}，请缩短 system prompt 或增大 CONTEXT_MAX_TOKENS",
                self.max_tokens
            );
        }
        let available = self.max_tokens - system_cost;
        let summary_chars =
            (available * 30 / 100).saturating_sub(SUMMARY_PREFIX.chars().count() + 8);

        if available < 256 || summary_chars < 64 || self.strategy == ContextStrategy::Truncate {
            let kept = keep_recent_turns(&turns, available);
            let kept = fit_last_turn(kept, available)?;
            let dropped_turns = turns.len().saturating_sub(kept.len());
            rebuild_history(history, system_messages, None, kept);
            log_compaction(
                self.strategy,
                original_estimate,
                estimate_messages(history),
                dropped_turns,
            );
            return Ok(());
        }

        let recent_ratio = match self.strategy {
            ContextStrategy::Summarize => 55,
            ContextStrategy::Layered => 45,
            ContextStrategy::Truncate => unreachable!(),
        };
        let recent_budget = available * recent_ratio / 100;
        let (older_turns, mut recent_turns) = split_recent_turns(&turns, recent_budget);

        let summary = if older_turns.is_empty() {
            previous_summary.chars().take(summary_chars).collect()
        } else {
            summarize_turns(
                client,
                self.strategy,
                &previous_summary,
                &older_turns,
                summary_chars,
            )
            .await?
        };

        while recent_turns.len() > 1
            && estimate_messages(&system_messages)
                + estimate_summary(&summary)
                + estimate_turns(&recent_turns)
                > self.max_tokens
        {
            recent_turns.remove(0);
        }
        let dropped_turns = turns.len().saturating_sub(recent_turns.len());
        let recent_turns = fit_last_turn(
            recent_turns,
            self.max_tokens
                .saturating_sub(system_cost + estimate_summary(&summary)),
        )?;
        rebuild_history(
            history,
            system_messages,
            (!summary.is_empty()).then_some(summary),
            recent_turns,
        );
        log_compaction(
            self.strategy,
            original_estimate,
            estimate_messages(history),
            dropped_turns,
        );
        Ok(())
    }

    pub fn clear(&self, history: &mut Vec<Message>) {
        history.retain(|message| message.role == "system" && !is_summary(message));
    }
}

fn is_summary(message: &Message) -> bool {
    message.role == "system"
        && message
            .content
            .as_deref()
            .is_some_and(|content| content.starts_with(SUMMARY_PREFIX))
}

fn split_turns(history: &[Message]) -> Vec<Vec<Message>> {
    let mut turns: Vec<Vec<Message>> = Vec::new();
    for message in history.iter().filter(|message| message.role != "system") {
        if message.role == "user" || turns.is_empty() {
            turns.push(Vec::new());
        }
        turns
            .last_mut()
            .expect("a turn was just created")
            .push(message.clone());
    }
    turns
}

fn rebuild_history(
    history: &mut Vec<Message>,
    system_messages: Vec<Message>,
    summary: Option<String>,
    turns: Vec<Vec<Message>>,
) {
    history.clear();
    history.extend(system_messages);
    if let Some(summary) = summary {
        history.push(Message::system(format!("{SUMMARY_PREFIX}{summary}")));
    }
    history.extend(turns.into_iter().flatten());
}

fn estimate_messages(messages: &[Message]) -> usize {
    messages.iter().map(estimate_message).sum()
}

fn estimate_message(message: &Message) -> usize {
    let content_chars = message
        .content
        .as_deref()
        .map(|content| content.chars().count())
        .unwrap_or_default();
    let calls_chars = message
        .tool_calls
        .as_ref()
        .map(|calls| {
            calls
                .iter()
                .map(|call| {
                    call.id.chars().count()
                        + call.function.name.chars().count()
                        + call.function.arguments.chars().count()
                        + 20
                })
                .sum::<usize>()
        })
        .unwrap_or_default();
    content_chars + calls_chars + message.role.len() + 8
}

fn estimate_summary(summary: &str) -> usize {
    if summary.is_empty() {
        0
    } else {
        summary.chars().count() + SUMMARY_PREFIX.chars().count() + 8
    }
}

fn estimate_turns(turns: &[Vec<Message>]) -> usize {
    turns.iter().map(|turn| estimate_messages(turn)).sum()
}

fn keep_recent_turns(turns: &[Vec<Message>], budget: usize) -> Vec<Vec<Message>> {
    let mut kept = Vec::new();
    let mut used = 0;
    for turn in turns.iter().rev() {
        let cost = estimate_messages(turn);
        if !kept.is_empty() && used + cost > budget {
            break;
        }
        kept.push(turn.clone());
        used += cost;
    }
    kept.reverse();
    kept
}

fn split_recent_turns(
    turns: &[Vec<Message>],
    budget: usize,
) -> (Vec<Vec<Message>>, Vec<Vec<Message>>) {
    let mut recent: Vec<Vec<Message>> = Vec::new();
    let mut used = 0;
    for (index, turn) in turns.iter().enumerate().rev() {
        let cost = estimate_messages(turn);
        if !recent.is_empty() && used + cost > budget {
            return (turns[..=index].to_vec(), recent.into_iter().rev().collect());
        }
        recent.push(turn.clone());
        used += cost;
    }
    (Vec::new(), recent.into_iter().rev().collect())
}

fn fit_last_turn(mut turns: Vec<Vec<Message>>, budget: usize) -> Result<Vec<Vec<Message>>> {
    if let Some(last_turn) = turns.last_mut() {
        let current_cost = estimate_messages(last_turn);
        if current_cost > budget {
            let mut excess = current_cost - budget;
            for message in last_turn.iter_mut() {
                if excess == 0 {
                    break;
                }
                let Some(content) = message.content.as_mut() else {
                    continue;
                };
                let content_len = content.chars().count();
                let removed = content_len.min(excess);
                if removed > 0 {
                    *content = content.chars().take(content_len - removed).collect();
                    excess -= removed;
                }
            }
            if excess > 0 {
                bail!("最新一轮的工具调用参数已超过上下文预算，无法安全截断");
            }
        }
    }
    Ok(turns)
}

async fn summarize_turns(
    client: &LlmClient,
    strategy: ContextStrategy,
    previous_summary: &str,
    turns: &[Vec<Message>],
    max_summary_chars: usize,
) -> Result<String> {
    let batch_chars = max_summary_chars.max(128);
    let mut batches: Vec<String> = Vec::new();
    let mut current = String::new();
    for turn in turns {
        let rendered = render_turn(turn);
        if !current.is_empty() && current.chars().count() + rendered.chars().count() > batch_chars {
            batches.push(std::mem::take(&mut current));
        }
        if rendered.chars().count() > batch_chars {
            if !current.is_empty() {
                batches.push(std::mem::take(&mut current));
            }
            let chars: Vec<_> = rendered.chars().collect();
            batches.extend(
                chars
                    .chunks(batch_chars)
                    .map(|chunk| chunk.iter().collect()),
            );
        } else {
            current.push_str(&rendered);
        }
    }
    if !current.is_empty() {
        batches.push(current);
    }

    let mut summary = previous_summary.to_string();
    for batch in batches {
        let instructions = match strategy {
            ContextStrategy::Summarize => {
                "用简洁中文更新对话摘要，保留用户事实、偏好、目标、决定和未解决事项。只写摘要，不要回答对话中的问题。"
            }
            ContextStrategy::Layered => {
                "用简洁中文更新分层记忆，只输出以下标题及必要内容：长期事实与偏好、当前目标与约束、已作出的决定、待处理事项。保留仍然有效的信息，删除过时内容。不要回答对话中的问题。"
            }
            ContextStrategy::Truncate => unreachable!(),
        };
        let prompt = format!(
            "{instructions}\n摘要长度不超过 {max_summary_chars} 个字符。\n\n已有摘要：\n{}\n\n需要整合的较早对话：\n{batch}",
            if summary.is_empty() {
                "（无）"
            } else {
                &summary
            }
        );
        summary = client
            .summarize(&prompt)
            .await
            .context("压缩较早对话失败")?;
        summary = summary.chars().take(max_summary_chars).collect();
    }
    Ok(summary)
}

fn render_turn(turn: &[Message]) -> String {
    let mut rendered = String::new();
    for message in turn {
        rendered.push_str(&format!(
            "{}: {}\n",
            message.role,
            message.content.as_deref().unwrap_or("")
        ));
        if let Some(calls) = &message.tool_calls {
            for call in calls {
                rendered.push_str(&format!(
                    "tool call {}({})\n",
                    call.function.name, call.function.arguments
                ));
            }
        }
    }
    rendered
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(role: &str, content: impl Into<String>) -> Message {
        Message::new(role, content)
    }

    #[test]
    fn parses_only_supported_strategies() {
        assert_eq!(
            ContextStrategy::parse("Truncate").unwrap(),
            ContextStrategy::Truncate
        );
        assert!(ContextStrategy::parse("unknown").is_err());
    }

    #[test]
    fn keeps_recent_complete_turns_when_truncating() {
        let turns = vec![
            vec![message("user", "old"), message("assistant", "old answer")],
            vec![message("user", "new"), message("assistant", "new answer")],
        ];
        let kept = keep_recent_turns(&turns, estimate_messages(&turns[1]));
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0][0].content.as_deref(), Some("new"));
    }

    #[test]
    fn trims_an_oversized_latest_turn_to_its_budget() {
        let turns = vec![vec![message("user", "a long question")]];
        let original_cost = estimate_turns(&turns);
        let fitted = fit_last_turn(turns, original_cost - 5).unwrap();
        assert!(estimate_turns(&fitted) <= original_cost - 5);
        assert_eq!(fitted[0][0].role, "user");
    }

    #[test]
    fn splits_history_without_separating_tool_results_from_their_turn() {
        let history = vec![
            message("system", "system"),
            message("user", "question"),
            Message::assistant(
                None,
                Some(vec![crate::llm::ToolCall {
                    id: "call_1".into(),
                    call_type: "function".into(),
                    function: crate::llm::FunctionCall {
                        name: "add".into(),
                        arguments: r#"{"a":1,"b":2}"#.into(),
                    },
                }]),
            ),
            Message::tool("call_1", "3"),
            message("assistant", "answer"),
        ];
        let turns = split_turns(&history);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].len(), 4);
    }

    #[test]
    fn clear_removes_managed_summary_but_keeps_system_prompt() {
        let manager = ContextManager::new(ContextStrategy::Layered, 1000);
        let mut history = vec![
            message("system", "system prompt"),
            message("system", format!("{SUMMARY_PREFIX}memory")),
            message("user", "question"),
        ];
        manager.clear(&mut history);
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].content.as_deref(), Some("system prompt"));
    }
}
