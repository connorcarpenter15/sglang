fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_root = std::path::PathBuf::from("../../proto");
    let proto_path = proto_root.join("sglang/runtime/v1/sglang.proto");
    let protoc_include = protoc_bin_vendored::include_path()?;
    let mut prost_config = prost_build::Config::new();
    prost_config.protoc_executable(protoc_bin_vendored::protoc_bin_path()?);

    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .protoc_arg("--experimental_allow_proto3_optional")
        .file_descriptor_set_path(
            std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap())
                .join("sglang_descriptor.bin"),
        )
        .compile_protos_with_config(
            prost_config,
            &[proto_path.clone()],
            &[proto_root, protoc_include],
        )?;

    println!("cargo:rerun-if-changed={}", proto_path.display());
    println!(
        "cargo:rerun-if-changed={}",
        "../../proto/sglang/runtime/v1/SCHEMA.sha256"
    );
    Ok(())
}
