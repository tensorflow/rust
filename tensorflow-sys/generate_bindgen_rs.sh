#!/bin/sh

if ! which bindgen > /dev/null; then
    echo "ERROR: Please install 'bindgen' using cargo:"
    echo "    cargo install bindgen"
    echo "See https://github.com/servo/rust-bindgen for more information."
    exit 1
fi

# See https://github.com/servo/rust-bindgen/issues/550 as to why
# this is blacklisted.
bindgen_options="--blacklist-type max_align_t"
header="/usr/include/tensorflow/c_api.h"

cmd="bindgen ${bindgen_options} ${header} --output src/bindgen.rs"
echo ${cmd}
${cmd}
