#!/bin/sh

if ! which bindgen > /dev/null; then
    echo "ERROR: Please install 'bindgen' using cargo:"
    echo "    cargo install bindgen"
    echo "See https://github.com/servo/rust-bindgen for more information."
    exit 1
fi

include_dir="$HOME/git/tensorflow"

bindgen_options_c_api="--allowlist-function TF_.+ --allowlist-type TF_.+ --allowlist-var TF_.+ --size_t-is-usize --default-enum-style=rust --generate-inline-functions"
cmd="bindgen ${bindgen_options_c_api} ${include_dir}/tensorflow/c/c_api.h --output src/c_api.rs --  -I ${include_dir}"
echo ${cmd}
${cmd}

bindgen_options_eager="--allowlist-function TFE_.+ --allowlist-type TFE_.+ --allowlist-var TFE_.+ --blocklist-type TF_.+ --size_t-is-usize --default-enum-style=rust --generate-inline-functions"
cmd="bindgen ${bindgen_options_eager} ${include_dir}/tensorflow/c/eager/c_api.h --output src/eager/c_api.rs --  -I ${include_dir}"
echo ${cmd}
${cmd}