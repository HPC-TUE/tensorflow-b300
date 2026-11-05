#!/bin/bash

# Builds the TensorFlow wheel, same steps as Dockerfile_compile but pinned and at runtime,
# so it can run as a non-root Run:ai job. The wheel ends up in /build/out.
# Runs inside the image from Containerfile_build, which holds the toolchain and bazelrc.

set -euxo pipefail

TF_COMMIT=5693b4a7feaff77f23af8a8218cd6e8e7aa2a7ca
BUILD_DIR=/build
CPUS=32
RAM_MB=110000

git init "$BUILD_DIR/tensorflow"
cd "$BUILD_DIR/tensorflow"
git remote add origin https://github.com/tensorflow/tensorflow.git
git fetch --depth 1 origin "$TF_COMMIT"
git checkout FETCH_HEAD

git diff --no-index --stat .bazelrc "$BUILD_DIR/bazelrc" || true
cp "$BUILD_DIR/bazelrc" .bazelrc

# cuda13_version has to stay last: cuda and cuda_nvcc both expand to cuda12_version
bazel build //tensorflow/tools/pip_package:wheel \
    --announce_rc \
    --repo_env=WHEEL_NAME=tensorflow \
    --local_cpu_resources=$CPUS --local_ram_resources=$RAM_MB --jobs=$CPUS \
    --config=cuda --config=cuda_nvcc --config=cuda_wheel --config=cuda13_version \
    --repo_env=HERMETIC_NCCL_VERSION=2.30.7

mkdir -p "$BUILD_DIR/out"
find -L bazel-bin/tensorflow/tools/pip_package -name '*.whl' -exec cp {} "$BUILD_DIR/out/" \;
ls -la "$BUILD_DIR/out"
echo BUILD_SUCCEEDED
