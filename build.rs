use std::path::{ Path, PathBuf };
use std::fs::File;
use std::io::Write;

const SHADERS: &[&str] = &[
    "src/test.comp.glsl",
];

fn write_binary_data<P>(path: P, data: &[u8])
    where P: AsRef<Path>
{
    let mut file = File::create(path)
        .expect("Failed to create file");

    file.write_all(data)
        .expect("Failed to write data");
}

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();

    let mut compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_include_callback(
        |a: &str, include_type: shaderc::IncludeType, b: &str, depth: usize| {
            panic!("B: {:?}", include_type);

            Err("Hehe".to_string())
        }
    );

    options.set_target_env(shaderc::TargetEnv::Vulkan,
                           shaderc::EnvVersion::Vulkan1_0 as u32);

    for shader in SHADERS {
        let t = Path::new(shader);

        let mut shader_output_path = PathBuf::from(&out_dir);
        shader_output_path.push(t.file_name().unwrap());
        shader_output_path.set_extension("spv");

        println!("cargo:rerun-if-changed={}", shader);
        let source = std::fs::read_to_string(shader)
            .expect("Failed to read shader source");
        let result = compiler.compile_into_spirv(&source,
                                                 shaderc::ShaderKind::Compute,
                                                 shader,
                                                 "main",
                                                 Some(&options))
            .expect("Failed to build shader");

        let name = shader_output_path.file_name()
            .expect("No file name");
        let name = name.to_str()
            .expect("Failed to convert to str");
        let shader_output_path = shader_output_path.as_path().to_str()
            .expect("Failed to convert to str");
        write_binary_data(shader_output_path, result.as_binary_u8());
        println!("cargo:rustc-env={}={}", name, shader_output_path);
    }

}
