# TensorFlow for NVIDIA B300 (CUDA 13, sm_103)

An unofficial TensorFlow build that runs on NVIDIA B300 GPUs (Blackwell Ultra, compute
capability 10.3). It is TensorFlow built from source for CUDA 13, with the recipe to rebuild it
and a GPU smoke test.

| | |
|---|---|
| TensorFlow | 2.23.0.dev0, commit `5693b4a7feaff77f23af8a8218cd6e8e7aa2a7ca` |
| Keras | 3.16 dev |
| Python | 3.12 |
| CUDA / cuDNN / NCCL | 13.0 / 9.12 / 2.30.7 |
| Tested on | NVIDIA B300 SXM6 AC, driver 580.173.02 |
| Result | 37/37 smoke test checks pass, including XLA ([log](logs/smoke_b300_release_1.0.log)) |

## Why this exists

A TensorFlow wheel built for CUDA 12 starts on the B300 but crashes as soon as it touches the GPU:

```
TensorFlow was not built with CUDA kernel binaries compatible with compute capability 10.3a.
gpuGetDeviceProperties(...): cudaErrorSharedObjectSymbolNotFound: shared object symbol not found
```

([full log](logs/smoke_b300_run1.log)). Building against CUDA 13 fixes this.

## Use the image

Download the wheel from the [Releases](../../releases) page into `tensorflow/`, then build the
image on an x86_64 machine (or with `--platform linux/amd64` on Apple silicon):

```bash
podman build --platform linux/amd64 -f Containerfile_release -t tensorflow-b300:1.0 .
```

The image runs as a non-root user (UID 2002) with no password, sudo or SSH. Started without a
command, it runs the smoke test:

```bash
podman run --rm --device nvidia.com/gpu=all tensorflow-b300:1.0
# or: docker run --rm --gpus all tensorflow-b300:1.0
```

A good run ends with `37/37 checks passed`. To run it explicitly and also fail on any GPU that
is not a B300:

```bash
python /gpu_smoke_test.py --expect-arch sm_103
```

On Kubernetes or Run:ai, a job that runs as another UID (for example to match your storage
permissions) works too. `HOME` is writable for any UID.

The wheel has only been tested inside this image, where CUDA comes from the NVIDIA base image.
Installing it elsewhere with `pip install "./tensorflow-2.23.0.dev0+selfbuilt-cp312-cp312-linux_x86_64.whl[and-cuda]"`
should pull CUDA 13 from NVIDIA's pip packages, but that has not been tested.

## Rebuild the wheel

The build runs as a non-root user, so it also works on clusters that don't allow root
containers.

1. Build the build image: `podman build --platform linux/amd64 -f Containerfile_build -t tf-b300-build .`
2. Start it with at least 32 cores and 128 GB of RAM (it idles with `sleep infinity`).
3. Inside it, run `/build/build_wheel.sh`. The wheel ends up in `/build/out`.

The build took about 70 minutes on 32 cores. Things that matter:

- `--config=cuda13_version` must come **after** the other CUDA configs. `cuda` and `cuda_nvcc`
  both expand to `cuda12_version`, and the last one wins.
- NCCL is pinned to 2.30.7 with `--repo_env=HERMETIC_NCCL_VERSION=2.30.7`.
- `tensorflow/.bazelrc` is the Bazel config used for this build. `build_wheel.sh` copies it over
  the one in the TensorFlow checkout and prints the difference.

## float32 precision

TensorFlow uses TF32 for float32 matmuls and convolutions on recent NVIDIA GPUs by default,
which trades some precision for speed. If you need full float32 precision, turn it off:

```python
tf.config.experimental.enable_tensor_float_32_execution(False)
```

## Files

| Path | What it is |
|---|---|
| `Containerfile_release` | Runtime image with the wheel and smoke test |
| `Containerfile_build`, `build_wheel.sh`, `install_bazelisk.sh` | Non-root wheel build |
| `tensorflow/.bazelrc` | Bazel config used for the build |
| `gpu_smoke_test.py` | GPU smoke test: cuBLAS, cuSolver, cuFFT, XLA, Keras layers and training |
| `logs/` | Test logs from the B300: old CUDA 12 wheel, new wheel, release image |

## License

The files in this repository are licensed under the Apache License 2.0. TensorFlow is Apache
2.0. NVIDIA components are not included and stay under NVIDIA's license. See [NOTICE](NOTICE).

This is not an official TensorFlow, Google or NVIDIA release.
