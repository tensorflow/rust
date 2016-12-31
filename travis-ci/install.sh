#!/bin/bash

BAZEL_URL=https://github.com/bazelbuild/bazel/releases/download/0.4.3/bazel-0.4.3-installer-linux-x86_64.sh

# Copied from https://raw.githubusercontent.com/michaelklishin/jdk_switcher/master/jdk_switcher.sh
source jdk_switcher.sh
jdk_switcher use oraclejdk8

set -e

curl -o bazel.sh -L ${BAZEL_URL}
chmod +x bazel.sh
./bazel.sh --user
