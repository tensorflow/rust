#!/bin/sh

if ! which bindgen > /dev/null; then
    echo "ERROR: Please install 'bindgen' using cargo:"
    echo "    cargo install bindgen"
    echo "See https://github.com/servo/rust-bindgen for more information."
    exit 1
fi

include_dir="$HOME/git/tensorflow"

bindgen_options_runtime_functions="--allowlist-function TF_.+ --blocklist-type .+ --size_t-is-usize --default-enum-style=rust --generate-inline-functions"
cmd="bindgen ${bindgen_options_runtime_functions} ${include_dir}/tensorflow/c/c_api.h --output src/runtime_linking/c_api.rs --  -I ${include_dir}"
echo ${cmd}
${cmd}

bindgen_options_runtime_types="--allowlist-type TF_.+ --blocklist-function .+ --size_t-is-usize --default-enum-style=rust --generate-inline-functions"
cmd="bindgen ${bindgen_options_runtime_types} ${include_dir}/tensorflow/c/c_api.h --output src/runtime_linking/types.rs --  -I ${include_dir}"
echo ${cmd}
${cmd}

echo "link! {\n$(cat src/runtime_linking/c_api.rs)" > src/runtime_linking/c_api.rs
echo } >> src/runtime_linking/c_api.rs
