#!/bin/sh

if ! which bindgen > /dev/null; then
    echo "ERROR: Please install 'bindgen' using cargo:"
    echo "    cargo install bindgen"
    echo "See https://github.com/servo/rust-bindgen for more information."
    exit 1
fi

include_dir="../../tensorflow"

# Export C-API
bindgen_options_c_api="--allowlist-function TF_.+ --allowlist-type TF_.+ --allowlist-var TF_.+ --size_t-is-usize --default-enum-style=rust --generate-inline-functions"
cmd="bindgen ${bindgen_options_c_api} ${include_dir}/tensorflow/c/c_api.h --output src/c_api.rs -- -I ${include_dir}"
echo ${cmd}
${cmd}

# Export PluggableDeviceLibrary from C-API experimental
bindgen_options_c_api_experimental="--allowlist-function TF_.+PluggableDeviceLibrary.* --blocklist-type TF_.+ --size_t-is-usize"
cmd="bindgen ${bindgen_options_c_api_experimental} ${include_dir}/tensorflow/c/c_api_experimental.h --output src/c_api_experimental.rs -- -I ${include_dir}"
echo ${cmd}
${cmd}

# Export Eager C-API
bindgen_options_eager="--allowlist-function TFE_.+ --allowlist-type TFE_.+ --allowlist-var TFE_.+ --blocklist-type TF_.+ --size_t-is-usize --default-enum-style=rust --generate-inline-functions --no-layout-tests"
cmd="bindgen ${bindgen_options_eager} ${include_dir}/tensorflow/c/eager/c_api.h --output src/eager/c_api.rs -- -I ${include_dir}"
echo ${cmd}
${cmd}

bindgen_options_experimental="--no-derive-copy --allowlist-function TF_LoadPluggableDeviceLibrary --allowlist-function TF_DeletePluggableDeviceLibraryHandle --allowlist-var TF_Buffer* --allowlist-type TF_ShapeAndTypeList --allowlist-type TF_ShapeAndType --allowlist-type TF_CheckpointReader --allowlist-type TF_AttrBuilder --size_t-is-usize --default-enum-style=rust --generate-inline-functions --blocklist-type TF_Library --blocklist-type TF_DataType --blocklist-type TF_Status"
cmd="bindgen ${bindgen_options_experimental} ${include_dir}/tensorflow/c/c_api_experimental.h --output src/experimental/c_api.rs --  -I ${include_dir}"
echo ${cmd}
${cmd}
