#!/bin/sh

if ! which bindgen > /dev/null; then
    echo "ERROR: Please install 'bindgen' using cargo:"
    echo "    cargo install bindgen"
    echo "See https://github.com/servo/rust-bindgen for more information."
    exit 1
fi

# See https://github.com/servo/rust-bindgen/issues/550 as to why
# this is blacklisted.
bindgen_options="--blacklist-type max_align_t --size_t-is-usize --default-enum-style=rust --generate-inline-functions"
# TODO: revert
include_dir="$HOME/git/tensorflow"

cmd="bindgen ${bindgen_options} ${include_dir}/tensorflow/c/c_api.h --output src/bindgen.rs --  -I ${include_dir}"
echo ${cmd}
${cmd}
