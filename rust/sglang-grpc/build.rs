fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_path = "../../proto/sglang/runtime/v1/sglang.proto";

    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .protoc_arg("--experimental_allow_proto3_optional")
        .file_descriptor_set_path(
            std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap())
                .join("sglang_descriptor.bin"),
        )
        .compile_protos(&[proto_path], &["../../proto"])?;

    // OpenEngine v1 — vendored copy synced from openengine/proto via openengine/gen.sh.
    // Server-only: this crate hosts the OpenEngine server bridged to the SGLang
    // scheduler; the Dynamo sidecar is the client.
    let openengine_proto = "proto/openengine.proto";
    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&[openengine_proto], &["proto"])?;

    println!("cargo:rerun-if-changed={}", proto_path);
    println!("cargo:rerun-if-changed={}", openengine_proto);
    Ok(())
}
