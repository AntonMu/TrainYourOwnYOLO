---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

Please understand that many of the models included in this repository are experimental and research-style code. If you open a GitHub issue, here is our policy:

1. It must be a bug, a feature request, or a significant problem with the documentation (for small docs fixes please send a PR instead).
2. The form below must be filled out.

------------------------

### System information
- **What is the top-level directory of the model you are using**:
- **Have I written custom code (as opposed to using a stock example script provided in the repo)**:
- **OS Platform and Distribution (e.g., Linux Ubuntu 16.04)**:
- **TensorFlow installed from (source or binary)**:
- **TensorFlow version (use command below)**:
- **Bazel version (if compiling from source)**:
- **CUDA/cuDNN version**:
- **GPU model and memory**:
- **Exact command to reproduce**:

You can obtain the TensorFlow version with

`python -c "import tensorflow as tf; print(tf.GIT_VERSION, tf.VERSION)"`

### Describe the problem
Describe the problem clearly here. Be sure to convey here why it's a bug or a feature request.

### Source code / logs
Include any logs or source code that would be helpful to diagnose the problem. If including tracebacks, please include the full traceback. Large logs and files should be attached. Try to provide a reproducible test case that is the bare minimum necessary to generate the problem.
