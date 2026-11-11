use serde_json::{json, Value};

/// 返回给 LLM 的工具 schema 列表
pub fn schemas() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "add",
                "description": "计算两个数字的和",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": { "type": "number", "description": "第一个加数" },
                        "b": { "type": "number", "description": "第二个加数" }
                    },
                    "required": ["a", "b"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "获取当前系统时间（本地时区，ISO 8601 格式）",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }),
    ]
}

/// 根据工具名和参数执行，返回字符串结果
pub fn execute(name: &str, args: &Value) -> String {
    match name {
        "add" => {
            let a = args.get("a").and_then(Value::as_f64);
            let b = args.get("b").and_then(Value::as_f64);
            match (a, b) {
                (Some(a), Some(b)) => {
                    // 整数结果不带小数点，更自然
                    let sum = a + b;
                    if sum.fract() == 0.0 {
                        format!("{}", sum as i64)
                    } else {
                        format!("{sum}")
                    }
                }

                _ => "错误：缺少参数 a 或 b".to_string(),
            }
        }
        "get_current_time" => {
            // 用 chrono 拿本地时间
            let now = chrono::Local::now();
            now.to_rfc3339()
        }
        other => format!("错误：未知工具 {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposes_both_tool_schemas() {
        let schemas = schemas();
        let names: Vec<_> = schemas
            .iter()
            .map(|schema| schema["function"]["name"].as_str().unwrap())
            .collect();

        assert_eq!(names, ["add", "get_current_time"]);
    }

    #[test]
    fn adds_two_numbers() {
        assert_eq!(execute("add", &json!({"a": 2, "b": 3})), "5");
        assert_eq!(execute("add", &json!({"a": 1.25, "b": 2.5})), "3.75");
    }

    #[test]
    fn returns_local_time_as_rfc3339() {
        let result = execute("get_current_time", &json!({}));
        assert!(chrono::DateTime::parse_from_rfc3339(&result).is_ok());
    }
}
