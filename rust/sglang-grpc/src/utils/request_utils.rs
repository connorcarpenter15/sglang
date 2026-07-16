use std::collections::HashMap;

use crate::proto;

fn prost_value_to_json(value: &prost_types::Value) -> serde_json::Value {
    use prost_types::value::Kind;
    match value.kind.as_ref() {
        None | Some(Kind::NullValue(_)) => serde_json::Value::Null,
        Some(Kind::BoolValue(value)) => serde_json::json!(value),
        Some(Kind::NumberValue(value)) => serde_json::json!(value),
        Some(Kind::StringValue(value)) => serde_json::json!(value),
        Some(Kind::ListValue(value)) => {
            serde_json::Value::Array(value.values.iter().map(prost_value_to_json).collect())
        }
        Some(Kind::StructValue(value)) => prost_struct_to_json(value),
    }
}

fn prost_struct_to_json(value: &prost_types::Struct) -> serde_json::Value {
    serde_json::Value::Object(
        value
            .fields
            .iter()
            .map(|(key, value)| (key.clone(), prost_value_to_json(value)))
            .collect(),
    )
}

fn regex_escape_literal(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        if matches!(
            character,
            '.' | '^' | '$' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '\\'
        ) {
            escaped.push('\\');
        }
        escaped.push(character);
    }
    escaped
}

fn sampling_params_to_map(params: &Option<proto::SamplingParams>) -> serde_json::Value {
    let Some(params) = params else {
        return serde_json::json!({});
    };
    let mut map = serde_json::Map::new();
    macro_rules! optional {
        ($field:ident) => {
            if let Some(value) = params.$field {
                map.insert(stringify!($field).into(), serde_json::json!(value));
            }
        };
    }
    optional!(temperature);
    optional!(top_p);
    optional!(top_k);
    optional!(min_p);
    optional!(frequency_penalty);
    optional!(presence_penalty);
    optional!(repetition_penalty);
    optional!(max_new_tokens);
    optional!(min_new_tokens);
    optional!(ignore_eos);
    optional!(n);
    if let Some(seed) = params.seed {
        map.insert("sampling_seed".into(), serde_json::json!(seed));
    }
    if let Some(max_thinking_tokens) = params.max_thinking_tokens {
        map.insert(
            "custom_params".into(),
            serde_json::json!({"thinking_budget": max_thinking_tokens}),
        );
    }
    let string_stops: Vec<_> = params
        .string_stops
        .iter()
        .map(|stop| stop.value.clone())
        .collect();
    if !string_stops.is_empty() {
        map.insert("stop".into(), serde_json::json!(string_stops));
    }
    let token_stops: Vec<_> = params
        .token_stops
        .iter()
        .map(|stop| stop.token_id)
        .collect();
    if !token_stops.is_empty() {
        map.insert("stop_token_ids".into(), serde_json::json!(token_stops));
    }
    if params
        .string_stops
        .iter()
        .any(|stop| stop.include_in_output)
        || params.token_stops.iter().any(|stop| stop.include_in_output)
    {
        map.insert("no_stop_trim".into(), serde_json::json!(true));
    }
    if let Some(guided) = params.guided_decoding.as_ref() {
        use proto::guided_decoding::Constraint;
        match guided.constraint.as_ref() {
            Some(Constraint::JsonSchema(value)) => {
                map.insert("json_schema".into(), serde_json::json!(value));
            }
            Some(Constraint::Regex(value)) => {
                map.insert("regex".into(), serde_json::json!(value));
            }
            Some(Constraint::Ebnf(value)) => {
                map.insert("ebnf".into(), serde_json::json!(value));
            }
            Some(Constraint::Choice(value)) => {
                let regex = value
                    .values
                    .iter()
                    .map(|choice| regex_escape_literal(choice))
                    .collect::<Vec<_>>()
                    .join("|");
                map.insert("regex".into(), serde_json::json!(format!("(?:{regex})")));
            }
            Some(Constraint::StructuralTag(value)) => {
                map.insert("structural_tag".into(), serde_json::json!(value));
            }
            None => {}
        }
        if let Some(backend) = guided.backend.as_ref() {
            map.insert("guided_decoding_backend".into(), serde_json::json!(backend));
        }
        if let Some(pattern) = guided.whitespace_pattern.as_ref() {
            map.insert(
                "guided_whitespace_pattern".into(),
                serde_json::json!(pattern),
            );
        }
    }
    serde_json::Value::Object(map)
}

fn trace_headers_to_json(headers: &HashMap<String, String>) -> Option<serde_json::Value> {
    (!headers.is_empty()).then(|| serde_json::json!(headers))
}

fn insert_disaggregated_params(
    request: &mut HashMap<String, serde_json::Value>,
    params: &Option<proto::DisaggregatedParams>,
) {
    let Some(params) = params else { return };
    request.insert(
        "bootstrap_host".into(),
        serde_json::json!(params.bootstrap_host),
    );
    request.insert(
        "bootstrap_port".into(),
        serde_json::json!(params.bootstrap_port),
    );
    request.insert(
        "bootstrap_room".into(),
        serde_json::json!(params.bootstrap_room),
    );
    if let Some(rank) = params.prefill_dp_rank {
        request.insert("disagg_prefill_dp_rank".into(), serde_json::json!(rank));
    }
    if let Some(pair_key) = params.bootstrap_pair_key.as_ref() {
        request.insert("bootstrap_pair_key".into(), serde_json::json!(pair_key));
    }
    if let Some(tp_size) = params.decode_tp_size {
        request.insert("decode_tp_size".into(), serde_json::json!(tp_size));
    }
}

fn tensor_element_size(dtype: proto::TensorDataType) -> Result<usize, String> {
    match dtype {
        proto::TensorDataType::Uint8 => Ok(1),
        proto::TensorDataType::Float16 | proto::TensorDataType::Bfloat16 => Ok(2),
        proto::TensorDataType::Int32 | proto::TensorDataType::Float32 => Ok(4),
        proto::TensorDataType::Int64 | proto::TensorDataType::Float64 => Ok(8),
        proto::TensorDataType::Unspecified => Err("tensor dtype must be specified".into()),
    }
}

fn validate_external_buffer(buffer: &proto::ExternalBuffer, minimum: usize) -> Result<(), String> {
    use proto::external_buffer::Transport;
    let Some(Transport::Nixl(nixl)) = buffer.transport.as_ref() else {
        return Err("external buffer requires a NIXL descriptor".into());
    };
    if nixl.metadata.is_empty() || nixl.descriptor.is_empty() {
        return Err("NIXL metadata and descriptor must not be empty".into());
    }
    if let Some(length) = nixl.length {
        let length = usize::try_from(length).map_err(|_| "NIXL length does not fit usize")?;
        if length < minimum {
            return Err(format!(
                "NIXL buffer length {length} is smaller than the required {minimum} bytes"
            ));
        }
    }
    Ok(())
}

fn validate_tensor(tensor: &proto::Tensor, require_shape: bool) -> Result<(), String> {
    use proto::tensor::Storage;
    if let Some(Storage::Serialized(serialized)) = tensor.storage.as_ref() {
        if serialized.format.is_empty() || serialized.data.is_empty() {
            return Err("serialized tensor requires a format and non-empty payload".into());
        }
        // Self-describing formats such as torch.save carry dtype and shape in
        // the payload. If callers also provide typed metadata, validate it
        // below and verify the shape again after deserialization in Python.
        if tensor.dtype == proto::TensorDataType::Unspecified as i32
            && tensor.shape.is_empty()
            && tensor.strides.is_empty()
        {
            return Ok(());
        }
    }
    let dtype = proto::TensorDataType::try_from(tensor.dtype)
        .map_err(|_| format!("unknown tensor dtype {}", tensor.dtype))?;
    let element_size = tensor_element_size(dtype)?;
    if require_shape && tensor.shape.is_empty() {
        return Err("tensor shape must not be empty".into());
    }
    if tensor.shape.iter().any(|dimension| *dimension <= 0) {
        return Err("tensor shape dimensions must be positive".into());
    }
    if !tensor.strides.is_empty() && tensor.strides.len() != tensor.shape.len() {
        return Err("tensor strides must be empty or match the shape rank".into());
    }
    if tensor.strides.iter().any(|stride| *stride < 0) {
        return Err("negative tensor strides are not supported".into());
    }
    let elements = if tensor.shape.is_empty() {
        0_usize
    } else if tensor.strides.is_empty() {
        tensor.shape.iter().try_fold(1_usize, |total, dimension| {
            total
                .checked_mul(usize::try_from(*dimension).map_err(|_| "invalid tensor shape")?)
                .ok_or("tensor shape size overflow")
        })?
    } else {
        tensor.shape.iter().zip(&tensor.strides).try_fold(
            1_usize,
            |offset, (dimension, stride)| {
                let dimension = usize::try_from(*dimension).map_err(|_| "invalid tensor shape")?;
                let stride = usize::try_from(*stride).map_err(|_| "invalid tensor stride")?;
                offset
                    .checked_add(
                        (dimension - 1)
                            .checked_mul(stride)
                            .ok_or("tensor stride overflow")?,
                    )
                    .ok_or("tensor stride overflow")
            },
        )?
    };
    let required_bytes = elements
        .checked_mul(element_size)
        .ok_or("tensor byte size overflow")?;
    match tensor.storage.as_ref() {
        Some(Storage::InlineData(data)) if data.len() == required_bytes => Ok(()),
        Some(Storage::InlineData(data)) => Err(format!(
            "inline tensor contains {} bytes but shape/dtype require {required_bytes}",
            data.len()
        )),
        Some(Storage::External(buffer)) => validate_external_buffer(buffer, required_bytes),
        Some(Storage::Serialized(_)) => Ok(()),
        None => Err("tensor storage must be specified".into()),
    }
}

fn validate_multimodal_inputs(inputs: &[proto::MultimodalInput]) -> Result<(), String> {
    use proto::multimodal_input::Source;
    let hash_count = inputs
        .iter()
        .filter(|input| input.routing_hash.is_some())
        .count();
    if hash_count != 0 && hash_count != inputs.len() {
        return Err(format!(
            "multimodal routing hashes must be supplied for every input ({hash_count}/{})",
            inputs.len()
        ));
    }
    for input in inputs {
        let modality = proto::Modality::try_from(input.modality)
            .map_err(|_| format!("unknown multimodal modality {}", input.modality))?;
        if modality == proto::Modality::Unspecified {
            return Err("multimodal modality must be specified".into());
        }
        match input.source.as_ref() {
            Some(Source::Url(url)) if !url.trim().is_empty() => {}
            Some(Source::Url(_)) => return Err("multimodal URL must not be empty".into()),
            Some(Source::InlineData(data)) if !data.is_empty() => {}
            Some(Source::InlineData(_)) => {
                return Err("inline multimodal data must not be empty".into());
            }
            Some(Source::DecodedTensor(tensor)) => validate_tensor(tensor, true)?,
            Some(Source::ExternalTensor(external)) => {
                let tensor = proto::Tensor {
                    dtype: external.dtype,
                    shape: external.shape.clone(),
                    strides: external.strides.clone(),
                    storage: external
                        .buffer
                        .clone()
                        .map(proto::tensor::Storage::External),
                };
                validate_tensor(&tensor, true)?;
            }
            None => return Err("multimodal input source must be specified".into()),
        }
    }
    Ok(())
}

fn tensor_to_json(tensor: &proto::Tensor) -> serde_json::Value {
    use proto::tensor::Storage;
    let storage = match tensor.storage.as_ref() {
        Some(Storage::InlineData(data)) => serde_json::json!({
            "kind": "inline",
            "data": data,
        }),
        Some(Storage::External(external)) => {
            use proto::external_buffer::Transport;
            match external.transport.as_ref() {
                Some(Transport::Nixl(nixl)) => serde_json::json!({
                    "kind": "nixl",
                    "metadata": nixl.metadata,
                    "descriptor": nixl.descriptor,
                    "agent_name": nixl.agent_name,
                    "length": nixl.length,
                }),
                None => serde_json::Value::Null,
            }
        }
        Some(Storage::Serialized(serialized)) => serde_json::json!({
            "kind": "serialized",
            "format": serialized.format,
            "data": serialized.data,
        }),
        None => serde_json::Value::Null,
    };
    serde_json::json!({
        "dtype": tensor.dtype,
        "shape": tensor.shape,
        "strides": tensor.strides,
        "storage": storage,
    })
}

fn multimodal_to_json(input: &proto::MultimodalInput) -> serde_json::Value {
    use proto::multimodal_input::Source;
    let source = match input.source.as_ref() {
        Some(Source::Url(value)) => serde_json::json!({"kind": "url", "value": value}),
        Some(Source::InlineData(value)) => {
            serde_json::json!({"kind": "inline", "value": value})
        }
        Some(Source::DecodedTensor(value)) => {
            serde_json::json!({"kind": "tensor", "value": tensor_to_json(value)})
        }
        Some(Source::ExternalTensor(value)) => {
            let tensor = proto::Tensor {
                dtype: value.dtype,
                shape: value.shape.clone(),
                strides: value.strides.clone(),
                storage: value.buffer.clone().map(proto::tensor::Storage::External),
            };
            serde_json::json!({"kind": "external", "value": tensor_to_json(&tensor)})
        }
        None => serde_json::Value::Null,
    };
    serde_json::json!({
        "modality": input.modality,
        "source": source,
        "mime_type": input.mime_type,
        "routing_hash": input.routing_hash,
    })
}

fn now_timestamp() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

pub(crate) fn extract_model_path(json_info: &str) -> String {
    serde_json::from_str::<serde_json::Value>(json_info)
        .ok()
        .and_then(|value| value.get("model_path")?.as_str().map(str::to_owned))
        .unwrap_or_default()
}

pub(crate) fn build_generate_dict(
    rid: &str,
    req: &proto::GenerateRequest,
) -> Result<HashMap<String, serde_json::Value>, String> {
    use proto::generate_request::Input;
    if let Some(params) = req.sampling_params.as_ref() {
        if params.n.is_some_and(|value| value <= 0) {
            return Err("n must be greater than zero".into());
        }
        if params.max_thinking_tokens.is_some_and(|value| value < 0) {
            return Err("max_thinking_tokens must not be negative".into());
        }
        if params.string_stops.iter().any(|stop| stop.value.is_empty()) {
            return Err("string stop values must not be empty".into());
        }
        if let Some(guided) = params.guided_decoding.as_ref() {
            use proto::guided_decoding::Constraint;
            match guided.constraint.as_ref() {
                Some(Constraint::Choice(choice)) if choice.values.is_empty() => {
                    return Err("guided choice must contain at least one value".into());
                }
                Some(Constraint::Choice(choice))
                    if choice.values.iter().any(|value| value.is_empty()) =>
                {
                    return Err("guided choice values must not be empty".into());
                }
                Some(_) => {}
                None => return Err("guided decoding constraint must be specified".into()),
            }
        }
    }
    let mut request = HashMap::new();
    request.insert("rid".into(), serde_json::json!(rid));
    match req.input.as_ref() {
        Some(Input::Text(text)) => {
            request.insert("text".into(), serde_json::json!(text));
        }
        Some(Input::InputIds(ids)) => {
            if ids.values.is_empty() {
                return Err("input_ids must not be empty".into());
            }
            request.insert("input_ids".into(), serde_json::json!(ids.values));
        }
        Some(Input::InputEmbeds(tensor)) => {
            validate_tensor(tensor, true)?;
            request.insert("grpc_input_embeds".into(), tensor_to_json(tensor));
        }
        None => return Err("Generate requires text, input_ids, or input_embeds".into()),
    }
    request.insert(
        "sampling_params".into(),
        sampling_params_to_map(&req.sampling_params),
    );
    request.insert("stream".into(), serde_json::json!(req.stream));
    if let Some(options) = req.logprob_options.as_ref() {
        request.insert(
            "return_logprob".into(),
            serde_json::json!(options.return_logprobs),
        );
        request.insert(
            "top_logprobs_num".into(),
            serde_json::json!(options.top_logprobs),
        );
        request.insert(
            "logprob_start_len".into(),
            serde_json::json!(options.prompt_logprob_start.unwrap_or(-1)),
        );
        request.insert(
            "token_ids_logprob".into(),
            serde_json::json!(options.token_ids),
        );
        request.insert(
            "return_text_in_logprobs".into(),
            serde_json::json!(options.return_text),
        );
        request.insert(
            "return_routed_experts".into(),
            serde_json::json!(options.return_routed_experts),
        );
        request.insert(
            "routed_experts_start_len".into(),
            serde_json::json!(options.routed_experts_start),
        );
        request.insert(
            "return_prompt_token_ids".into(),
            serde_json::json!(options.return_prompt_token_ids),
        );
    }
    if !req.multimodal_inputs.is_empty() {
        validate_multimodal_inputs(&req.multimodal_inputs)?;
        request.insert(
            "grpc_multimodal_inputs".into(),
            serde_json::Value::Array(
                req.multimodal_inputs
                    .iter()
                    .map(multimodal_to_json)
                    .collect(),
            ),
        );
    }
    if let Some(options) = req.multimodal_processor_options.as_ref() {
        request.insert(
            "grpc_multimodal_processor_options".into(),
            prost_struct_to_json(options),
        );
    }
    request.insert(
        "use_audio_in_video".into(),
        serde_json::json!(req.use_audio_in_video),
    );
    request.insert("priority".into(), serde_json::json!(req.priority));
    if let Some(params) = req.sampling_params.as_ref() {
        request.insert(
            "require_reasoning".into(),
            serde_json::json!(params.require_reasoning.unwrap_or(false)),
        );
        request.insert(
            "grpc_stop_visibility".into(),
            serde_json::json!({
                "strings": params.string_stops.iter().map(|stop| serde_json::json!({
                    "value": stop.value,
                    "include_in_output": stop.include_in_output,
                })).collect::<Vec<_>>(),
                "tokens": params.token_stops.iter().map(|stop| serde_json::json!({
                    "token_id": stop.token_id,
                    "include_in_output": stop.include_in_output,
                })).collect::<Vec<_>>(),
            }),
        );
    }
    for (name, value) in [
        ("lora_path", req.lora_path.as_ref()),
        ("lora_id", req.lora_id.as_ref()),
        ("routing_key", req.routing_key.as_ref()),
        ("session_id", req.session_id.as_ref()),
    ] {
        if let Some(value) = value {
            request.insert(name.into(), serde_json::json!(value));
        }
    }
    if let Some(rank) = req.routed_dp_rank {
        request.insert("routed_dp_rank".into(), serde_json::json!(rank));
    }
    insert_disaggregated_params(&mut request, &req.disaggregated_params);
    if let Some(trace) = trace_headers_to_json(&req.trace_headers) {
        request.insert("external_trace_header".into(), trace);
    }
    request.insert("received_time".into(), serde_json::json!(now_timestamp()));
    Ok(request)
}

pub(crate) fn embed_request_ids(rid: &str, input_count: usize) -> Vec<String> {
    if input_count == 1 {
        vec![rid.to_string()]
    } else {
        (0..input_count)
            .map(|index| format!("{rid}:{index}"))
            .collect()
    }
}

pub(crate) fn build_embed_dict(
    rid: &str,
    req: &proto::EmbedRequest,
) -> Result<HashMap<String, serde_json::Value>, String> {
    use proto::embed_input::Input;
    if req.inputs.is_empty() {
        return Err("Embed requires at least one input".into());
    }
    let mut texts = Vec::new();
    let mut token_batches = Vec::new();
    let mut tensors = Vec::new();
    for input in &req.inputs {
        match input.input.as_ref() {
            Some(Input::Text(value)) if token_batches.is_empty() && tensors.is_empty() => {
                texts.push(value.clone())
            }
            Some(Input::InputIds(value)) if texts.is_empty() && tensors.is_empty() => {
                token_batches.push(value.values.clone())
            }
            Some(Input::InputEmbeds(value)) if texts.is_empty() && token_batches.is_empty() => {
                validate_tensor(value, true)?;
                tensors.push(tensor_to_json(value))
            }
            Some(_) => return Err("Embed batch inputs must use one input representation".into()),
            None => return Err("Embed input is missing its value".into()),
        }
    }
    let mut request = HashMap::new();
    let request_ids = embed_request_ids(rid, req.inputs.len());
    if request_ids.len() == 1 {
        request.insert("rid".into(), serde_json::json!(request_ids[0]));
    } else {
        request.insert("rid".into(), serde_json::json!(request_ids));
    }
    if !texts.is_empty() {
        request.insert("text".into(), serde_json::json!(texts));
    } else if !token_batches.is_empty() {
        request.insert("input_ids".into(), serde_json::json!(token_batches));
    } else {
        request.insert("grpc_input_embeds".into(), serde_json::json!(tensors));
    }
    if let Some(dimensions) = req.dimensions {
        if dimensions == 0 {
            return Err("embedding dimensions must be greater than zero".into());
        }
        request.insert("dimensions".into(), serde_json::json!(dimensions));
    }
    request.insert("priority".into(), serde_json::json!(req.priority));
    if let Some(routing_key) = req.routing_key.as_ref() {
        request.insert("routing_key".into(), serde_json::json!(routing_key));
    }
    if let Some(trace) = trace_headers_to_json(&req.trace_headers) {
        request.insert("external_trace_header".into(), trace);
    }
    request.insert("received_time".into(), serde_json::json!(now_timestamp()));
    Ok(request)
}

pub(crate) fn build_classify_dict(
    rid: &str,
    req: &proto::ClassifyRequest,
) -> HashMap<String, serde_json::Value> {
    let mut request = HashMap::new();
    request.insert("rid".into(), serde_json::json!(rid));
    if !req.text.is_empty() {
        request.insert("text".into(), serde_json::json!(req.text));
    }
    if !req.input_ids.is_empty() {
        request.insert("input_ids".into(), serde_json::json!(req.input_ids));
    }
    if let Some(routing_key) = req.routing_key.as_ref() {
        request.insert("routing_key".into(), serde_json::json!(routing_key));
    }
    if let Some(trace) = trace_headers_to_json(&req.trace_headers) {
        request.insert("external_trace_header".into(), trace);
    }
    request.insert("received_time".into(), serde_json::json!(now_timestamp()));
    request
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_maps_choice_and_seed() {
        let request = proto::GenerateRequest {
            input: Some(proto::generate_request::Input::InputIds(proto::TokenIds {
                values: vec![1, 2],
            })),
            sampling_params: Some(proto::SamplingParams {
                seed: Some(7),
                guided_decoding: Some(proto::GuidedDecoding {
                    constraint: Some(proto::guided_decoding::Constraint::Choice(
                        proto::ChoiceConstraint {
                            values: vec!["a".into(), "b".into()],
                        },
                    )),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mapped = build_generate_dict("r", &request).unwrap();
        assert_eq!(mapped["input_ids"], serde_json::json!([1, 2]));
        assert_eq!(mapped["sampling_params"]["sampling_seed"], 7);
        assert_eq!(mapped["sampling_params"]["regex"], "(?:a|b)");
    }

    #[test]
    fn generate_escapes_guided_choices_as_regex() {
        let request = proto::GenerateRequest {
            input: Some(proto::generate_request::Input::InputIds(proto::TokenIds {
                values: vec![1],
            })),
            sampling_params: Some(proto::SamplingParams {
                guided_decoding: Some(proto::GuidedDecoding {
                    constraint: Some(proto::guided_decoding::Constraint::Choice(
                        proto::ChoiceConstraint {
                            values: vec!["a+b".into(), "x.y".into()],
                        },
                    )),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mapped = build_generate_dict("r", &request).unwrap();
        assert_eq!(mapped["sampling_params"]["regex"], "(?:a\\+b|x\\.y)");
    }

    #[test]
    fn generate_maps_limits_and_logprobs() {
        let request = proto::GenerateRequest {
            input: Some(proto::generate_request::Input::Text("hello".into())),
            sampling_params: Some(proto::SamplingParams {
                max_new_tokens: Some(8),
                min_new_tokens: Some(8),
                ignore_eos: Some(true),
                ..Default::default()
            }),
            logprob_options: Some(proto::LogprobOptions {
                return_logprobs: true,
                top_logprobs: 2,
                ..Default::default()
            }),
            ..Default::default()
        };

        let mapped = build_generate_dict("r", &request).unwrap();
        assert_eq!(mapped["sampling_params"]["max_new_tokens"], 8);
        assert_eq!(mapped["sampling_params"]["min_new_tokens"], 8);
        assert_eq!(mapped["sampling_params"]["ignore_eos"], true);
        assert_eq!(mapped["return_logprob"], true);
        assert_eq!(mapped["top_logprobs_num"], 2);
    }

    #[test]
    fn tensor_validation_rejects_shape_size_and_dtype_mismatches() {
        let mut tensor = proto::Tensor {
            dtype: proto::TensorDataType::Float32 as i32,
            shape: vec![2, 2],
            storage: Some(proto::tensor::Storage::InlineData(vec![0; 16])),
            ..Default::default()
        };
        assert!(validate_tensor(&tensor, true).is_ok());
        tensor.shape = vec![3, 2];
        assert!(validate_tensor(&tensor, true).is_err());
        tensor.shape = vec![2, 2];
        tensor.dtype = proto::TensorDataType::Unspecified as i32;
        assert!(validate_tensor(&tensor, true).is_err());
    }

    #[test]
    fn external_buffer_requires_descriptor_and_sufficient_length() {
        let mut tensor = proto::Tensor {
            dtype: proto::TensorDataType::Float32 as i32,
            shape: vec![2, 2],
            storage: Some(proto::tensor::Storage::External(proto::ExternalBuffer {
                transport: Some(proto::external_buffer::Transport::Nixl(proto::NixlBuffer {
                    metadata: b"metadata".to_vec(),
                    descriptor: b"descriptor".to_vec(),
                    length: Some(15),
                    ..Default::default()
                })),
            })),
            ..Default::default()
        };
        assert!(validate_tensor(&tensor, true).is_err());
        if let Some(proto::tensor::Storage::External(buffer)) = tensor.storage.as_mut()
            && let Some(proto::external_buffer::Transport::Nixl(nixl)) = buffer.transport.as_mut()
        {
            nixl.length = Some(16);
        }
        assert!(validate_tensor(&tensor, true).is_ok());
    }

    #[test]
    fn self_describing_serialized_tensor_may_omit_typed_metadata() {
        let tensor = proto::Tensor {
            dtype: proto::TensorDataType::Unspecified as i32,
            shape: Vec::new(),
            strides: Vec::new(),
            storage: Some(proto::tensor::Storage::Serialized(
                proto::SerializedTensor {
                    format: "pytorch".into(),
                    data: vec![1, 2, 3],
                },
            )),
        };
        assert!(validate_tensor(&tensor, true).is_ok());
    }

    #[test]
    fn multimodal_hashes_are_all_or_none() {
        let inputs = vec![
            proto::MultimodalInput {
                modality: proto::Modality::Image as i32,
                source: Some(proto::multimodal_input::Source::Url("https://a".into())),
                routing_hash: Some("hash-a".into()),
                ..Default::default()
            },
            proto::MultimodalInput {
                modality: proto::Modality::Image as i32,
                source: Some(proto::multimodal_input::Source::Url("https://b".into())),
                ..Default::default()
            },
        ];
        assert!(validate_multimodal_inputs(&inputs).is_err());
    }

    #[test]
    fn generate_rejects_invalid_sampling_contract() {
        let request = proto::GenerateRequest {
            input: Some(proto::generate_request::Input::Text("hello".into())),
            sampling_params: Some(proto::SamplingParams {
                n: Some(0),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(build_generate_dict("r", &request).is_err());
    }

    #[test]
    fn embed_assigns_a_unique_request_id_to_each_batch_item() {
        let request = proto::EmbedRequest {
            inputs: vec![
                proto::EmbedInput {
                    input: Some(proto::embed_input::Input::Text("one".into())),
                },
                proto::EmbedInput {
                    input: Some(proto::embed_input::Input::Text("two".into())),
                },
            ],
            ..Default::default()
        };

        let mapped = build_embed_dict("batch", &request).unwrap();
        assert_eq!(mapped["rid"], serde_json::json!(["batch:0", "batch:1"]));
        assert_eq!(mapped["text"], serde_json::json!(["one", "two"]));
    }
}
