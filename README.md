# rust-agent

用 Rust 写的最小 LLM Agent Demo：读取问题 → 调用大模型 → 打印回答。
基于 OpenAI 兼容的 `/chat/completions` 协议，可对接 OpenAI、DeepSeek、通义、
Moonshot、Ollama 等服务。

## 快速开始

要求 Rust 1.85+（依赖里有使用 edition 2024 的 crate）。

```bash
# 1. 配置密钥
cp .env.example .env
# 编辑 .env，至少填 LLM_API_KEY（以及需要的 LLM_BASE_URL / LLM_MODEL）

# 2. 运行（单次提问）
cargo run -- "用一句话解释什么是所有权"

# 3. 交互式多轮对话
cargo run
```

交互模式下支持：

- `/clear` 清空上下文（保留 system 提示词）
- `/exit` 退出（或 Ctrl-D）
- Windows 中文控制台（代码页 936 / GBK）输入已做兼容，无需手动 `chcp 65001`

## 配置项

| 环境变量 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `LLM_API_KEY` | 是 | — | 也兼容 `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` |
| `LLM_BASE_URL` | 否 | `https://api.openai.com/v1` | 需实现 `/chat/completions` |
| `LLM_MODEL` | 否 | `gpt-4o-mini` | 模型名 |
| `LLM_SYSTEM_PROMPT` | 否 | 中文助手提示词 | 置空可关闭 system 消息 |
| `LLM_TEMPERATURE` | 否 | `0.7` | 采样温度 |
| `LLM_TIMEOUT_SECS` | 否 | `120` | 请求超时 |
| `LLM_STREAM` | 否 | `false` | `true` 开启流式输出，`false` 使用完整响应后输出 |

常见服务商示例：

```bash
# DeepSeek
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# 本地 Ollama（key 随便填一个非空值）
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:7b
LLM_API_KEY=ollama
```

## 代码结构

```
src/
├── main.rs    # 入口：参数解析、交互循环、维护对话历史、控制台编码兼容
├── config.rs  # 环境变量配置与默认值
├── llm.rs     # OpenAI 兼容客户端、消息/响应类型、单元测试
└── tools.rs   # 工具 schema 与执行函数
```

- `LlmClient::chat` 是唯一的网络出口，按配置选择普通 JSON 或 SSE 响应，返回
  `ChatOutput`（内容、工具调用、`finish_reason`、可用时的 token 用量）。
- Agent 暴露 `add`（两数相加）和 `get_current_time`（获取本地 ISO 8601 时间）工具；
  模型发起调用后，执行结果会以 `role: "tool"` 消息回填，并继续请求模型生成答复。
- 默认使用非流式响应：服务端返回完整答案后再显示。设置 `LLM_STREAM=true` 可启用
  SSE 流式响应，文本会随生成逐段显示；两种模式都支持工具调用，流式模式会合并工具参数片段后再执行。
- 请求失败时不会污染对话历史，便于重试。

| 模式 | 响应方式 | 用户体验 | 本项目配置 |
| --- | --- | --- | --- |
| 非流式 | 等模型完整生成后一次返回 | 实现简单，但等待期间看不到进度 | 默认，或 `LLM_STREAM=false` |
| 流式 | 服务端以 SSE 分段发送内容 | 首字更快显示，可逐步看到回答 | `LLM_STREAM=true` |

## 测试

```bash
cargo test        # 覆盖响应解析、空内容、错误分支
cargo clippy --all-targets
```

## 后续可以扩展

1. **重试与限流**：对 429/5xx 做指数退避。
2. **会话持久化**：把 `history` 序列化到磁盘，支持恢复会话。
