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
macro_rules! log {
    ($fmt:expr) => (println!(concat!("libtensorflow-sys/build.rs:{}: ", $fmt), line!()));
    ($fmt:expr, $($arg:tt)*) => (println!(concat!("libtensorflow-sys/build.rs:{}: ", $fmt), line!(), $($arg)*));
}
macro_rules! log_var(($var:ident) => (log!(concat!(stringify!($var), " = {:?}"), $var)));

fn main() {
    if pkg_config::find_library(LIBRARY).is_ok() {
        log!("Returning early because {} was already found", LIBRARY);
        return;
    }

    let output = PathBuf::from(&get!("OUT_DIR"));
    log_var!(output);
    let source = PathBuf::from(&get!("CARGO_MANIFEST_DIR")).join(format!("target/source-{}", VERSION));
    log_var!(source);
    let lib_dir = output.join(format!("lib-{}", VERSION));
    log_var!(lib_dir);
    if lib_dir.exists() {
        log!("Directory {:?} already exists", lib_dir);
    } else {
        log!("Creating directory {:?}", lib_dir);
        fs::create_dir(lib_dir.clone()).unwrap();
    }
    let library_path = lib_dir.join(format!("lib{}.so", LIBRARY));
    log_var!(library_path);
    if library_path.exists() {
        log!("{:?} already exists, not building", library_path);
    } else {
        let target_path = &TARGET.replace(":", "/");
        log_var!(target_path);
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
        let target_bazel_bin = source.join("bazel-bin").join(target_path);
        log!("Copying {:?} to {:?}", target_bazel_bin, library_path);
        fs::copy(target_bazel_bin, library_path).unwrap();
    }

    println!("cargo:rustc-link-lib=dylib={}", LIBRARY);
    println!("cargo:rustc-link-search={}", lib_dir.display());
}

// Patches in https://github.com/tensorflow/tensorflow/commit/d03f2545ecc3012e6c941a3a1e957f7d7f8d5040
// (or close enough) to work around #18 (No C API in TensorFlow v0.10)
// TODO(acrume): Remove once we're on a non-broken version.
fn patch_build_file(source: &PathBuf) -> Result<(), Box<Error>> {
    let build_file = source.join("tensorflow/BUILD");
    log!("Checking build file {:?}", build_file);
    let mut content = String::new();
    let mut file = try!(OpenOptions::new().read(true).append(true).open(build_file.clone()));
    try!(file.read_to_string(&mut content));
    if content.contains("libtensorflow_c.so") {
        log!("Build file {:?} already patched", build_file);
    } else {
        log!("Patching build file {:?}", build_file);
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
    let configured = configure(&mut command);
    log!("Executing {:?}", configured);
    if !ok!(configured.status()).success() {
        panic!("failed to execute {:?}", configured);
    }
    log!("Command {:?} finished successfully", configured);
}
