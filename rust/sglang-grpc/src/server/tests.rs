use super::{
    DEFAULT_GRPC_MAX_MESSAGE_SIZE, generate_response_from_data, has_finish_reason,
    openai_status_code, resolve_max_message_size, terminal_error_status,
};

#[test]
fn frozen_schema_hashes_match_the_intentional_v1_baseline() {
    use sha2::{Digest, Sha256};

    let baseline = include_str!("../../../../proto/sglang/runtime/v1/SCHEMA.sha256");
    let value = |name: &str| {
        baseline
            .lines()
            .find_map(|line| line.strip_prefix(&format!("{name}=")))
            .unwrap_or_else(|| panic!("missing {name} in SCHEMA.sha256"))
    };
    let source = include_bytes!("../../../../proto/sglang/runtime/v1/sglang.proto");
    let source_hash = format!("{:x}", Sha256::digest(source));
    assert_eq!(source_hash, value("source_sha256"));
    assert_eq!(crate::descriptor_sha256(), value("descriptor_sha256"));
    assert_eq!(crate::PROTOCOL_REVISION, value("protocol_revision"));
}
use crate::bridge::{ResponseData, TerminalError};
use std::collections::HashMap;
use tonic::Code;

#[test]
fn openai_status_code_uses_forwarded_status_when_present() {
    let meta_info = HashMap::from([(String::from("status_code"), String::from("429"))]);
    assert_eq!(openai_status_code(&meta_info, 200), 429);
}

#[test]
fn openai_status_code_falls_back_when_missing_or_invalid() {
    assert_eq!(openai_status_code(&HashMap::new(), 200), 200);

    let meta_info = HashMap::from([(String::from("status_code"), String::from("not-an-int"))]);
    assert_eq!(openai_status_code(&meta_info, 200), 200);
}

#[test]
fn terminal_error_status_maps_channel_full_to_resource_exhausted() {
    let status = terminal_error_status(TerminalError::ChannelFull {
        rid: "rid".to_string(),
    });

    assert_eq!(status.code(), Code::ResourceExhausted);
}

#[test]
fn terminal_error_status_maps_abort_to_cancelled() {
    let status = terminal_error_status(TerminalError::Aborted {
        rid: "rid".to_string(),
    });

    assert_eq!(status.code(), Code::Cancelled);
}

#[test]
fn null_finish_reason_is_not_terminal() {
    let null_finish = HashMap::from([("finish_reason".to_string(), "null".to_string())]);
    assert!(!has_finish_reason(&null_finish));

    let stop_finish = HashMap::from([(
        "finish_reason".to_string(),
        r#"{"type":"stop","matched":null}"#.to_string(),
    )]);
    assert!(has_finish_reason(&stop_finish));
}

#[test]
fn cumulative_choice_output_is_converted_to_deltas() {
    let mut token_offset = 0;
    let mut text_offset = 0;
    let mut logprob_offset = 0;
    let first = generate_response_from_data(
        ResponseData {
            text: Some("he".into()),
            output_ids: Some(vec![1, 2]),
            embedding: None,
            embeddings: None,
            choice_index: 1,
            json_bytes: None,
            meta_info: HashMap::new(),
        },
        false,
        None,
        &mut token_offset,
        &mut text_offset,
        &mut logprob_offset,
    );
    let second = generate_response_from_data(
        ResponseData {
            text: Some("hello".into()),
            output_ids: Some(vec![1, 2, 3]),
            embedding: None,
            embeddings: None,
            choice_index: 1,
            json_bytes: None,
            meta_info: HashMap::new(),
        },
        false,
        None,
        &mut token_offset,
        &mut text_offset,
        &mut logprob_offset,
    );
    assert_eq!(first.delta_output_ids, vec![1, 2]);
    assert_eq!(first.delta_text.as_deref(), Some("he"));
    assert_eq!(second.delta_output_ids, vec![3]);
    assert_eq!(second.delta_text.as_deref(), Some("llo"));
}

#[test]
fn cumulative_logprobs_are_converted_to_deltas() {
    let response_data = |entries: &str, top_entries: &str| ResponseData {
        text: None,
        output_ids: None,
        embedding: None,
        embeddings: None,
        choice_index: 0,
        json_bytes: None,
        meta_info: HashMap::from([
            ("output_token_logprobs".to_string(), entries.to_string()),
            ("output_top_logprobs".to_string(), top_entries.to_string()),
        ]),
    };
    let mut token_offset = 0;
    let mut text_offset = 0;
    let mut logprob_offset = 0;

    let first = generate_response_from_data(
        response_data(
            "[[-0.1, 10, \"a\"], [-0.2, 11, \"b\"]]",
            "[[[-0.1, 10, \"a\"]], [[-0.2, 11, \"b\"]]]",
        ),
        false,
        None,
        &mut token_offset,
        &mut text_offset,
        &mut logprob_offset,
    );
    let second = generate_response_from_data(
        response_data(
            "[[-0.1, 10, \"a\"], [-0.2, 11, \"b\"], [-0.3, 12, \"c\"]]",
            "[[[-0.1, 10, \"a\"]], [[-0.2, 11, \"b\"]], [[-0.3, 12, \"c\"]]]",
        ),
        false,
        None,
        &mut token_offset,
        &mut text_offset,
        &mut logprob_offset,
    );

    let first_logprobs = first.logprobs.expect("first logprobs");
    assert_eq!(first_logprobs.output.len(), 2);
    assert_eq!(first_logprobs.output[0].token_id, 10);
    assert_eq!(first_logprobs.output[0].top_logprobs.len(), 1);

    let second_logprobs = second.logprobs.expect("second logprobs");
    assert_eq!(second_logprobs.output.len(), 1);
    assert_eq!(second_logprobs.output[0].token_id, 12);
    assert_eq!(second_logprobs.output[0].text.as_deref(), Some("c"));
}

// SAFETY: env vars are process-global; bundle all SGLANG_TONIC_PAYLOAD cases into one
// serial test so they don't race each other under `cargo test`'s default parallelism.
#[test]
fn resolve_max_message_size_honors_env_var() {
    const VAR: &str = "SGLANG_TONIC_PAYLOAD";

    // Unset → default.
    // SAFETY: single-threaded test mutating process env (see note above).
    unsafe {
        std::env::remove_var(VAR);
    }
    assert_eq!(resolve_max_message_size(), DEFAULT_GRPC_MAX_MESSAGE_SIZE);

    // Valid override → honored verbatim.
    unsafe {
        std::env::set_var(VAR, "1048576");
    }
    assert_eq!(resolve_max_message_size(), 1_048_576);

    // Invalid string → warn + fall back to default.
    unsafe {
        std::env::set_var(VAR, "not-a-number");
    }
    assert_eq!(resolve_max_message_size(), DEFAULT_GRPC_MAX_MESSAGE_SIZE);

    // Zero → treated as invalid, fall back to default.
    unsafe {
        std::env::set_var(VAR, "0");
    }
    assert_eq!(resolve_max_message_size(), DEFAULT_GRPC_MAX_MESSAGE_SIZE);

    unsafe {
        std::env::remove_var(VAR);
    }
}
