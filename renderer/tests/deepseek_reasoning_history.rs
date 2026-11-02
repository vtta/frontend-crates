// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use dynamo_protocols::types::CreateChatCompletionRequest;
use dynamo_renderer::{
    OAIPromptFormatter,
    deepseek::{
        common::ThinkingMode,
        v4::{self, DeepSeekV4Formatter},
        v32::DeepSeekV32Formatter,
        v41::{self, DeepSeekV41Formatter},
    },
};
use serde_json::{Value, json};

fn history(reasoning: Value) -> Value {
    json!({
        "model": "deepseek-v4.1",
        "messages": [
            {"role": "user", "content": "Inspect both files."},
            {"role": "assistant", "content": "", "reasoning_content": reasoning,
             "tool_calls": [
                 {"id": "a", "type": "function", "function": {"name": "read_a", "arguments": "{}"}},
                 {"id": "b", "type": "function", "function": {"name": "read_b", "arguments": "{}"}}
             ]},
            {"role": "tool", "tool_call_id": "a", "content": "file A"},
            {"role": "tool", "tool_call_id": "b", "content": "file B"}
        ],
        "tools": [{"type": "function", "function": {
            "name": "read_a", "parameters": {"type": "object", "properties": {}}
        }}]
    })
}

fn formatters() -> [Box<dyn OAIPromptFormatter>; 3] {
    [
        Box::new(DeepSeekV32Formatter::new_thinking()),
        Box::new(DeepSeekV4Formatter::new_thinking()),
        Box::new(DeepSeekV41Formatter),
    ]
}

#[test]
fn segmented_reasoning_matches_native_flat_history() {
    for formatter in formatters() {
        for (segments, flat) in [
            (
                json!(["Keep the prior plan.", "", ""]),
                "Keep the prior plan.",
            ),
            (json!(["first", "", "last"]), "first\nlast"),
            (
                json!(["'quotes' \\slashes\n中文", "next", ""]),
                "'quotes' \\slashes\n中文\nnext",
            ),
        ] {
            let request: CreateChatCompletionRequest =
                serde_json::from_value(history(segments.clone())).unwrap();
            let before = serde_json::to_value(&request).unwrap();
            let flat_request: CreateChatCompletionRequest =
                serde_json::from_value(history(json!(flat))).unwrap();
            let actual = formatter.render(&request).unwrap();
            assert_eq!(actual, formatter.render(&flat_request).unwrap());
            assert!(actual.contains(flat));
            assert!(!actual.contains(&segments.to_string()));
            assert_eq!(serde_json::to_value(&request).unwrap(), before);
        }
    }
}

#[test]
fn empty_segments_match_missing_reasoning() {
    for formatter in formatters() {
        let empty: CreateChatCompletionRequest =
            serde_json::from_value(history(json!(["", "", ""]))).unwrap();
        let mut absent = history(Value::Null);
        absent["messages"][1]
            .as_object_mut()
            .unwrap()
            .remove("reasoning_content");
        let absent: CreateChatCompletionRequest = serde_json::from_value(absent).unwrap();
        assert_eq!(
            formatter.render(&empty).unwrap(),
            formatter.render(&absent).unwrap()
        );
    }
}

#[test]
fn earlier_turn_reasoning_is_still_dropped_without_tools() {
    for formatter in formatters() {
        for reasoning in [json!("OLD_REASONING"), json!(["OLD_REASONING", ""])] {
            let request: CreateChatCompletionRequest = serde_json::from_value(json!({
                "model": "deepseek-v4.1",
                "messages": [
                    {"role": "user", "content": "First question"},
                    {"role": "assistant", "content": "First answer", "reasoning_content": reasoning},
                    {"role": "user", "content": "Next question"}
                ]
            })).unwrap();
            assert!(
                !formatter
                    .render(&request)
                    .unwrap()
                    .contains("OLD_REASONING")
            );
        }
    }
}

#[test]
fn public_v4_encoder_preserves_segmented_reasoning() {
    let mut messages = history(json!(["first", "", "last"]))["messages"]
        .as_array()
        .unwrap()
        .clone();
    let actual =
        v4::encode_messages_with_options(&messages, ThinkingMode::Thinking, true, false, None)
            .unwrap();
    messages[1]["reasoning_content"] = json!("first\nlast");
    assert_eq!(
        actual,
        v4::encode_messages_with_options(&messages, ThinkingMode::Thinking, true, false, None,)
            .unwrap()
    );
}

#[test]
fn v41_rejects_literal_image_placeholders_in_reasoning_segments() {
    for reasoning in [
        json!("<｜deepseek_image｜>"),
        json!(["", "<｜deepseek_image｜>", ""]),
    ] {
        let messages = history(reasoning)["messages"].as_array().unwrap().clone();
        assert!(v41::encode_messages(&messages, ThinkingMode::Thinking, false, 75).is_err());
    }
}
