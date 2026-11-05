//! 最小 LLM Agent Demo
//!
//! 用法：
//!   cargo run -- "用一句话解释什么是 Rust"
//!   cargo run            # 进入交互式多轮对话

mod config;
mod llm;

use std::io::{self, BufRead, Write};

use anyhow::Result;

use config::Config;
use llm::{ChatOutput, LlmClient, Message};

const HELP: &str = "\
rust-agent — 最小 LLM 调用 Demo

用法:
  rust-agent                 进入交互式多轮对话
  rust-agent <问题>          单次提问并退出
  rust-agent --help          显示帮助

配置（环境变量或 .env 文件）:
  LLM_API_KEY        必填，也可使用 OPENAI_API_KEY / DEEPSEEK_API_KEY
  LLM_BASE_URL       接口地址，默认 https://api.openai.com/v1
  LLM_MODEL          模型名，默认 gpt-4o-mini
  LLM_SYSTEM_PROMPT  系统提示词，置空可关闭
  LLM_TEMPERATURE    采样温度，默认 0.7
  LLM_TIMEOUT_SECS   请求超时秒数，默认 120

交互模式命令:
  /clear  清空对话上下文（保留 system 提示词）
  /exit   退出（也可用 Ctrl-D）
";

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{HELP}");
        return Ok(());
    }

    let config = Config::from_env()?;
    let client = LlmClient::new(&config)?;

    // 对话历史：Agent 后续要做工具调用循环时，也是围绕这份历史展开。
    let mut history: Vec<Message> = Vec::new();
    if let Some(prompt) = &config.system_prompt {
        history.push(Message::system(prompt));
    }

    eprintln!(
        "[config] model={} endpoint={}/chat/completions",
        config.model,
        config.base_url.trim_end_matches('/')
    );

    // 单次提问模式
    if !args.is_empty() {
        let question = args.join(" ");
        let output = ask(&client, &mut history, &question).await?;
        println!("{}", output.content);
        return Ok(());
    }

    // 交互模式
    eprintln!("输入问题开始对话；/clear 清空上下文，/exit 或 Ctrl-D 退出。");

    loop {
        print!("\n你 > ");
        io::stdout().flush().ok();

        let Some(line) = read_line()? else {
            println!();
            break;
        };

        match line.trim() {
            "" => continue,
            "/exit" | "/quit" => break,
            "/clear" => {
                history.retain(|m| m.role == "system");
                eprintln!("[context] 已清空对话上下文");
                continue;
            }
            question => {
                match ask(&client, &mut history, question).await {
                    Ok(output) => println!("\n助手 > {}", output.content),
                    Err(err) => eprintln!("\n[错误] {err:#}"),
                }
            }
        }
    }

    Ok(())
}

/// 把问题追加进历史、调用 LLM、再把回答写回历史。
/// 调用失败时回滚用户消息，保证历史仍然自洽。
async fn ask(client: &LlmClient, history: &mut Vec<Message>, question: &str) -> Result<ChatOutput> {
    history.push(Message::user(question));

    let output = match client.chat(history).await {
        Ok(output) => output,
        Err(err) => {
            history.pop();
            return Err(err);
        }
    };

    history.push(Message::assistant(output.content.clone()));

    if let Some(usage) = output.usage {
        eprintln!(
            "[tokens] prompt={} completion={} total={}",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        );
    }
    if let Some(reason) = &output.finish_reason {
        if reason == "length" {
            eprintln!("[warn] 回答因 max_tokens 被截断");
        }
    }

    Ok(output)
}

/// 读取一行用户输入。
///
/// 中文 Windows 控制台默认代码页是 936（GBK），直接 `read_line` 会因字节不是
/// 合法 UTF-8 而报错，所以这里先按字节读取，再做 UTF-8 → GBK 的兼容解码。
fn read_line() -> Result<Option<String>> {
    let mut buf = Vec::new();
    if io::stdin().lock().read_until(b'\n', &mut buf)? == 0 {
        return Ok(None);
    }
    Ok(Some(decode_line(&buf)))
}

fn decode_line(bytes: &[u8]) -> String {
    match std::str::from_utf8(bytes) {
        Ok(text) => text.to_string(),
        Err(_) => encoding_rs::GBK.decode(bytes).0.into_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_utf8_line() {
        assert_eq!(decode_line("你好\n".as_bytes()).trim(), "你好");
    }

    #[test]
    fn decodes_gbk_line() {
        // “中文” 在 GBK 下的字节
        assert_eq!(decode_line(&[0xD6, 0xD0, 0xCE, 0xC4, b'\n']).trim(), "中文");
    }
}
