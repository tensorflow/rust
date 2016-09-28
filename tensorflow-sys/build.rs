extern crate pkg_config;

use std::error::Error;
use std::fs::OpenOptions;
use std::io::Read;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

const LIBRARY: &'static str = "tensorflow_c";
const REPOSITORY: &'static str = "https://github.com/tensorflow/tensorflow.git";
const TARGET: &'static str = "tensorflow:libtensorflow_c.so";
const VERSION: &'static str = "0.10.0";

macro_rules! get(($name:expr) => (ok!(env::var($name))));
macro_rules! ok(($expression:expr) => ($expression.unwrap()));

fn main() {
    if pkg_config::find_library(LIBRARY).is_ok() {
        return;
    }

    let target_path = &TARGET.replace(":", "/");
    let output = PathBuf::from(&get!("OUT_DIR"));
    if !output.join(target_path).exists() {
        let source = PathBuf::from(&get!("CARGO_MANIFEST_DIR")).join(format!("target/source-{}", VERSION));
        if !Path::new(&source.join(".git")).exists() {
            run("git", |command| command.arg("clone")
                                        .arg(format!("--branch=v{}", VERSION))
                                        .arg("--recursive")
                                        .arg(REPOSITORY)
                                        .arg(&source));
        }
        patch_build_file(&source).unwrap();
        run("./configure", |command| command.current_dir(&source));
        run("bazel", |command| command.current_dir(&source)
                                      .arg("build")
                                      .arg(format!("--jobs={}", get!("NUM_JOBS")))
                                      .arg("--compilation_mode=opt")
                                      .arg(TARGET));
        let library_path = output.join(format!("{}.so", LIBRARY));
        if library_path.exists() {
            // The file gets copied as readonly, so we get a permission error when overwriting an existing copy.
            // The simplest way to solve it is to delete the old file, if it exists.
            if let Err(e) = fs::remove_file(&library_path) {
                println!("cargo:warning=Unable to delete old copy of {}: {}",
                         library_path.to_string_lossy(), e)
            }
        }
        fs::copy(source.join("bazel-bin").join(target_path), library_path).unwrap();
    }

    println!("cargo:rustc-link-lib=dylib={}", LIBRARY);
    println!("cargo:rustc-link-search={}", output.display());
}

// Patches in https://github.com/tensorflow/tensorflow/commit/d03f2545ecc3012e6c941a3a1e957f7d7f8d5040
// (or close enough) to work around #18 (No C API in TensorFlow v0.10)
// TODO(acrume): Remove once we're on a non-broken version.
fn patch_build_file(source: &PathBuf) -> Result<(), Box<Error>> {
    let build_file = source.join("tensorflow/BUILD");
    let mut content = String::new();
    let mut file = try!(OpenOptions::new().read(true).append(true).open(build_file));
    try!(file.read_to_string(&mut content));
    if !content.contains("libtensorflow_c.so") {
        try!(file.write_all(br#"
cc_binary(
    name = "libtensorflow_c.so",
    linkshared = 1,
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/core:tensorflow",
    ],
)
"#));
    }
    Ok(())
}

fn run<F>(name: &str, mut configure: F) where F: FnMut(&mut Command) -> &mut Command {
    let mut command = Command::new(name);
    if !ok!(configure(&mut command).status()).success() {
        panic!("failed to execute {:?}", command);
    }
}
