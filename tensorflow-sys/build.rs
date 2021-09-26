extern crate curl;
extern crate flate2;
extern crate pkg_config;
extern crate semver;
extern crate tar;

use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{self, Command};

use curl::easy::Easy;
use flate2::read::GzDecoder;
use semver::Version;
use tar::Archive;
use zip::ZipArchive;

const FRAMEWORK_LIBRARY: &str = "tensorflow_framework";
const LIBRARY: &str = "tensorflow";
const REPOSITORY: &str = "https://github.com/tensorflow/tensorflow.git";
const FRAMEWORK_TARGET: &str = "tensorflow:libtensorflow_framework";
const TARGET: &str = "tensorflow:libtensorflow";
// `VERSION` and `TAG` are separate because the tag is not always `'v' + VERSION`.
const VERSION: &str = "2.6.0";
const TAG: &str = "v2.6.0";
const MIN_BAZEL: &str = "3.7.2";

macro_rules! get(($name:expr) => (ok!(env::var($name))));
macro_rules! ok(($expression:expr) => ($expression.unwrap()));
macro_rules! log {
    ($fmt:expr) => (println!(concat!("tensorflow-sys/build.rs:{}: ", $fmt), line!()));
    ($fmt:expr, $($arg:tt)*) => (println!(concat!("tensorflow-sys/build.rs:{}: ", $fmt),
    line!(), $($arg)*));
}
macro_rules! log_var(($var:ident) => (log!(concat!(stringify!($var), " = {:?}"), $var)));

fn main() {
    // DO NOT RELY ON THIS
    if cfg!(feature = "private-docs-rs") {
        log!("Returning early because private-docs-rs feature was enabled");
        return;
    }

    if check_windows_lib() {
        log!("Returning early because {} was already found", LIBRARY);
        return;
    }

    // Note that pkg_config will print cargo:rustc-link-lib and cargo:rustc-link-search as
    // appropriate if the library is found.
    if pkg_config::probe_library(LIBRARY).is_ok() {
        log!("Returning early because {} was already found", LIBRARY);
        return;
    }

    let force_src = match env::var("TF_RUST_BUILD_FROM_SRC") {
        Ok(s) => s == "true",
        Err(_) => false,
    };

    let target_os = target_os();
    if !force_src
        && target_arch() == "x86_64"
        && (target_os == "linux" || target_os == "macos" || target_os == "windows")
    {
        install_prebuilt();
    } else {
        build_from_src();
    }
}

fn target_arch() -> String {
    get!("CARGO_CFG_TARGET_ARCH")
}

fn target_os() -> String {
    get!("CARGO_CFG_TARGET_OS")
}

fn dll_prefix() -> &'static str {
    match &target_os() as &str {
        "windows" => "",
        _ => "lib",
    }
}

fn dll_suffix() -> &'static str {
    match &target_os() as &str {
        "windows" => ".dll",
        "macos" => ".dylib",
        _ => ".so",
    }
}

fn check_windows_lib() -> bool {
    if target_os() != "windows" {
        return false;
    }
    let windows_lib: &str = &format!("{}.lib", LIBRARY);
    if let Ok(path) = env::var("PATH") {
        for p in path.split(';') {
            let path = Path::new(p).join(windows_lib);
            if path.exists() {
                println!("cargo:rustc-link-lib=dylib={}", LIBRARY);
                println!("cargo:rustc-link-search=native={}", p);
                return true;
            }
        }
    }
    false
}

fn remove_suffix(value: &mut String, suffix: &str) {
    if value.ends_with(suffix) {
        let n = value.len();
        value.truncate(n - suffix.len());
    }
}

fn has_extension<P: AsRef<Path>>(path: P, extension: &str) -> bool {
    if let Some(os_ext) = path.as_ref().extension() {
        if let Some(ext) = os_ext.to_str() {
            ext == extension
        } else {
            false
        }
    } else {
        false
    }
}

fn extract_tar_gz<P: AsRef<Path>, P2: AsRef<Path>>(archive_path: P, extract_to: P2) {
    let file = File::open(archive_path).unwrap();
    let unzipped = GzDecoder::new(file);
    let mut a = Archive::new(unzipped);
    a.unpack(extract_to).unwrap();
}

// NOTE: It's possible for this function to extract the dll and interface
//       file directly to OUT_DIR instead of extracting then making a
//       copy of the libraries in OUT_DIR.
//       The same approach could be utilized for the other implementation
//       of `extract`.
fn extract_zip<P: AsRef<Path>, P2: AsRef<Path>>(archive_path: P, extract_to: P2) {
    fs::create_dir_all(&extract_to).expect("Failed to create output path for zip archive.");
    let file = File::open(archive_path).expect("Unable to open libtensorflow zip archive.");
    let mut archive = ZipArchive::new(file).unwrap();
    for i in 0..archive.len() {
        let mut zipfile = archive.by_index(i).unwrap();
        let output_path = extract_to.as_ref().join(zipfile.sanitized_name());
        if zipfile.name().starts_with("lib") {
            if zipfile.is_dir() {
                fs::create_dir_all(&output_path)
                    .expect("Failed to create output directory when unpacking archive.");
            } else {
                if let Some(parent) = output_path.parent() {
                    if !parent.exists() {
                        fs::create_dir_all(&parent)
                            .expect("Failed to create parent directory for extracted file.");
                    }
                }
                let mut outfile = File::create(&output_path).unwrap();
                io::copy(&mut zipfile, &mut outfile).unwrap();
            }
        }
    }
}

fn extract<P: AsRef<Path>, P2: AsRef<Path>>(archive_path: P, extract_to: P2) {
    if has_extension(&archive_path, "zip") {
        extract_zip(archive_path, extract_to);
    } else {
        extract_tar_gz(archive_path, extract_to);
    }
}

// Downloads and unpacks a prebuilt binary. Only works for certain platforms.
fn install_prebuilt() {
    // Figure out the file names.
    let os = match &target_os() as &str {
        "macos" => "darwin".to_string(),
        x => x.to_string(),
    };
    let proc_type = if cfg!(feature = "tensorflow_gpu") {
        "gpu"
    } else {
        "cpu"
    };
    let windows = target_os() == "windows";
    let ext = if windows { ".zip" } else { ".tar.gz" };
    let binary_url = format!(
        "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-{}-{}-{}-{}{}",
        proc_type,
        os,
        target_arch(),
        VERSION,
        ext
    );
    log_var!(binary_url);
    let short_file_name = binary_url.split('/').last().unwrap();
    let mut base_name = short_file_name.to_string();
    remove_suffix(&mut base_name, ext);
    log_var!(base_name);
    let download_dir = match env::var("TF_RUST_DOWNLOAD_DIR") {
        Ok(s) => PathBuf::from(s),
        Err(_) => PathBuf::from(&get!("OUT_DIR")),
    };
    if !download_dir.exists() {
        fs::create_dir(&download_dir).unwrap();
    }
    let file_name = download_dir.join(short_file_name);
    log_var!(file_name);

    // Download the tarball.
    if !file_name.exists() {
        let f = File::create(&file_name).unwrap();
        let mut writer = BufWriter::new(f);
        let mut easy = Easy::new();
        easy.url(&binary_url).unwrap();
        easy.write_function(move |data| Ok(writer.write(data).unwrap()))
            .unwrap();
        easy.perform().unwrap();

        let response_code = easy.response_code().unwrap();
        if response_code != 200 {
            panic!(
                "Unexpected response code {} for {}",
                response_code, binary_url
            );
        }
    }

    // Extract the tarball.
    let unpacked_dir = download_dir.join(base_name);
    let lib_dir = unpacked_dir.join("lib");
    let framework_library_file = format!("{}{}{}", dll_prefix(), FRAMEWORK_LIBRARY, dll_suffix());
    let library_file = format!("{}{}{}", dll_prefix(), LIBRARY, dll_suffix());

    let framework_library_full_path = lib_dir.join(&framework_library_file);
    let library_full_path = lib_dir.join(&library_file);

    let download_required =
        (!windows && !framework_library_full_path.exists()) || !library_full_path.exists();

    if download_required {
        extract(file_name, &unpacked_dir);
    }

    if target_os() != "windows" {
        // There is no tensorflow_framework.dll
        println!("cargo:rustc-link-lib=dylib={}", FRAMEWORK_LIBRARY);
    }
    println!("cargo:rustc-link-lib=dylib={}", LIBRARY);
    let output = PathBuf::from(&get!("OUT_DIR"));

    // NOTE: The following shouldn't strictly be necessary. See note above `extract`.
    let framework_files = std::fs::read_dir(lib_dir).unwrap();
    for library_entry in framework_files.filter_map(Result::ok) {
        let library_full_path = library_entry.path();
        let new_library_full_path = output.join(&library_full_path.file_name().unwrap());
        if new_library_full_path.exists() {
            log!(
                "{} already exists. Removing",
                new_library_full_path.display()
            );
            fs::remove_file(&new_library_full_path).unwrap();
        }
        log!(
            "Copying {} to {}...",
            library_full_path.display(),
            new_library_full_path.display()
        );
        fs::copy(&library_full_path, &new_library_full_path).unwrap();
    }
    println!("cargo:rustc-link-search={}", output.display());
}

fn symlink<P: AsRef<Path>, P2: AsRef<Path>>(target: P, link: P2) {
    if link.as_ref().exists() {
        // Avoid errors if it already exists.
        fs::remove_file(link.as_ref()).unwrap();
    }
    log!(
        "Creating symlink {:?} pointing to {:?}",
        link.as_ref(),
        target.as_ref()
    );
    #[cfg(target_os = "windows")]
    std::os::windows::fs::symlink_file(target, link).unwrap();
    #[cfg(not(target_os = "windows"))]
    std::os::unix::fs::symlink(target, link).unwrap();
}

fn build_from_src() {
    let dll_suffix = dll_suffix();
    let framework_target = FRAMEWORK_TARGET.to_string() + dll_suffix;
    let target = TARGET.to_string() + dll_suffix;

    let output = PathBuf::from(&get!("OUT_DIR"));
    log_var!(output);
    let source = PathBuf::from(&get!("CARGO_MANIFEST_DIR")).join(format!("target/source-{}", TAG));
    log_var!(source);
    let lib_dir = output.join(format!("lib-{}", TAG));
    log_var!(lib_dir);
    if lib_dir.exists() {
        log!("Directory {:?} already exists", lib_dir);
    } else {
        log!("Creating directory {:?}", lib_dir);
        fs::create_dir(lib_dir.clone()).unwrap();
    }
    let framework_unversioned_library_path = lib_dir.join(format!("lib{}.so", FRAMEWORK_LIBRARY));
    let framework_library_path = lib_dir.join(format!("lib{}.so.2", FRAMEWORK_LIBRARY));
    log_var!(framework_library_path);
    let unversioned_library_path = lib_dir.join(format!("lib{}.so", LIBRARY));
    let library_path = lib_dir.join(format!("lib{}.so.2", LIBRARY));
    log_var!(library_path);
    if library_path.exists() && framework_library_path.exists() {
        log!(
            "{:?} and {:?} already exist, not building",
            library_path,
            framework_library_path
        );
    } else {
        if let Err(e) = check_bazel() {
            println!(
                "cargo:error=Bazel must be installed at version {} or greater. (Error: {})",
                MIN_BAZEL, e
            );
            process::exit(1);
        }
        let framework_target_path = &format!("{}.2", framework_target.replace(":", "/"));
        log_var!(framework_target_path);
        let target_path = &format!("{}.so", TARGET.replace(":", "/"));
        log_var!(target_path);
        if !Path::new(&source.join(".git")).exists() {
            run("git", |command| {
                command
                    .arg("clone")
                    .arg(format!("--branch={}", TAG))
                    .arg("--recursive")
                    .arg(REPOSITORY)
                    .arg(&source)
            });
        }
        // Only configure if not previously configured.  Configuring runs a
        // `bazel clean`, which we don't want, because we want to be able to
        // continue from a cancelled build.
        let configure_hint_file_pb = source.join(".rust-configured");
        let configure_hint_file = Path::new(&configure_hint_file_pb);
        if !configure_hint_file.exists() {
            run("bash", |command| {
                command
                    .current_dir(&source)
                    .env(
                        "TF_NEED_CUDA",
                        if cfg!(feature = "tensorflow_gpu") {
                            "1"
                        } else {
                            "0"
                        },
                    )
                    .arg("-c")
                    .arg("yes ''|./configure")
            });
            File::create(configure_hint_file).unwrap();
        }
        // Allows us to pass in --incompatible_load_argument_is_label=false
        // to work around https://github.com/tensorflow/tensorflow/issues/15492
        let bazel_args_string = if let Ok(args) = env::var("TF_RUST_BAZEL_OPTS") {
            args
        } else {
            "".to_string()
        };
        run("bazel", |command| {
            command
                .current_dir(&source)
                .arg("build")
                .arg(format!("--jobs={}", get!("NUM_JOBS")))
                .arg("--compilation_mode=opt")
                .arg("--copt=-march=native")
                .args(bazel_args_string.split_whitespace())
                .arg(&target)
        });
        let framework_target_bazel_bin = source.join("bazel-bin").join(framework_target_path);
        log!(
            "Copying {:?} to {:?}",
            framework_target_bazel_bin,
            framework_library_path
        );
        if framework_library_path.exists() {
            fs::remove_file(&framework_library_path).unwrap();
        }
        fs::copy(framework_target_bazel_bin, &framework_library_path).unwrap();
        let target_bazel_bin = source.join("bazel-bin").join(target_path);
        log!("Copying {:?} to {:?}", target_bazel_bin, library_path);
        if library_path.exists() {
            fs::remove_file(&library_path).unwrap();
        }
        fs::copy(target_bazel_bin, &library_path).unwrap();
    }
    symlink(
        framework_library_path.file_name().unwrap(),
        framework_unversioned_library_path,
    );
    symlink(library_path.file_name().unwrap(), unversioned_library_path);
    println!("cargo:rustc-link-lib=dylib={}", FRAMEWORK_LIBRARY);
    println!("cargo:rustc-link-lib=dylib={}", LIBRARY);
    println!("cargo:rustc-link-search={}", lib_dir.display());
}

fn run<F>(name: &str, mut configure: F)
where
    F: FnMut(&mut Command) -> &mut Command,
{
    let mut command = Command::new(name);
    let configured = configure(&mut command);
    log!("Executing {:?}", configured);
    if !ok!(configured.status()).success() {
        panic!("failed to execute {:?}", configured);
    }
    log!("Command {:?} finished successfully", configured);
}

// Building TF 0.11.0rc1 with Bazel 0.3.0 gives this error when running `configure`:
//   expected ConfigurationTransition or NoneType for 'cfg' while calling label_list but got
// string instead:     data.
//       ERROR: com.google.devtools.build.lib.packages.BuildFileContainsErrorsException: error
// loading package '': Extension file 'tensorflow/tensorflow.bzl' has errors.
// And the simple solution is to require Bazel 0.3.1 or higher.
fn check_bazel() -> Result<(), Box<dyn Error>> {
    let mut command = Command::new("bazel");
    command.arg("version");
    log!("Executing {:?}", command);
    let out = command.output()?;
    log!("Command {:?} finished successfully", command);
    let stdout = String::from_utf8(out.stdout)?;
    let mut found_version = false;
    for line in stdout.lines() {
        if line.starts_with("Build label:") {
            found_version = true;
            let mut version_str = line
                .split(':')
                .nth(1)
                .unwrap()
                .split(' ')
                .nth(1)
                .unwrap()
                .trim();
            if version_str.ends_with('-') {
                // hyphen is 1 byte long, so it's safe
                version_str = &version_str[..version_str.len() - 1];
            }
            let version = Version::parse(version_str)?;
            let want = Version::parse(MIN_BAZEL)?;
            if version < want {
                return Err(format!(
                    "Installed version {} is less than required version {}",
                    version_str, MIN_BAZEL
                )
                .into());
            }
        }
    }
    if !found_version {
        return Err("Did not find version number in `bazel version` output.".into());
    }
    Ok(())
}
