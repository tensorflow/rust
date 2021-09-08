#!/bin/sh

if ! which bindgen > /dev/null; then
    echo "ERROR: Please install 'bindgen' using cargo:"
    echo "    cargo install bindgen"
    echo "See https://github.com/servo/rust-bindgen for more information."
    exit 1
fi

# TODO: revert
include_dir="$HOME/git/tensorflow"

# Generate bindings for function, type, and var if its name starts with `TF_` or `TFE_`.
allowlist="--allowlist-function TFE?_.+ --allowlist-type TFE?_.+ --allowlist-var TFE?_.+"

# See https://github.com/servo/rust-bindgen/issues/550 as to why
bindgen_options="--size_t-is-usize --default-enum-style=rust --generate-inline-functions"

cmd="bindgen ${allowlist} ${bindgen_options} wrapper.h --output src/bindgen.rs -- -I ${include_dir}"
echo ${cmd}
${cmd}
