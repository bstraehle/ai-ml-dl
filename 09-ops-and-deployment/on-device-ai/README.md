---
status: current
last_reviewed: 2026-11-22
review_every: 12mo
---

# On-device AI

Running models on phones, laptops and edge devices instead of in the cloud, for lower latency, offline use, privacy and lower serving cost.

**Roles:** ML engineer, mobile and embedded developer.
**Prerequisites:** [Deep learning](../../01-foundations/deep-learning/), [Quantization](../../03-adapting-models/quantization/)

## Key concepts

- **Why on device:** responses without a network round trip, data that never leaves the device, and no per-request server cost.
- **Capture and compile.** Export a trained model to a portable format, then compile it for the target chip and runtime (for example LiteRT, formerly TensorFlow Lite, or ONNX Runtime) ([L3](L3_Student.ipynb)).
- **Test on real hardware.** Device farms such as Qualcomm AI Hub run your model on actual phones and report latency and memory ([L2](L2_Student.ipynb)).
- **Quantize for the NPU.** Integer models run much faster on phone neural processors; calibration data keeps accuracy close to the float model ([L4](L4_Student.ipynb)).
- **Delegates** hand the work to the GPU or NPU instead of the CPU ([appendix](Appendix-Building_the_App.ipynb)).
- **LLMs on device.** Small quantized LLMs run locally with tools such as llama.cpp and ExecuTorch.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Introduction to on-device AI](https://www.deeplearning.ai/short-courses/introduction-to-on-device-ai/) (Qualcomm) | Short course | Intermediate | Free when reviewed |
| [Qualcomm AI Hub](https://aihub.qualcomm.com/) | Platform | Intermediate | Free tier |
| [LiteRT](https://developers.google.com/edge/litert) (Google) | Runtime | Intermediate | Free |
| [ExecuTorch](https://docs.pytorch.org/executorch/) (PyTorch) | Runtime | Advanced | Free |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Runtime | Intermediate | Free |

## Hands-on

| Lab | What it covers |
|---|---|
| [L2_Student.ipynb](L2_Student.ipynb) | Deploying a segmentation model and running it on a real phone |
| [L3_Student.ipynb](L3_Student.ipynb) | Capturing and compiling a model for different runtimes |
| [L4_Student.ipynb](L4_Student.ipynb) | Quantizing a model with calibration data |
| [Appendix-Building_the_App.ipynb](Appendix-Building_the_App.ipynb) | Building the app with TensorFlow Lite and the Qualcomm delegate |

## Next steps

- [Serving LLMs efficiently](../efficiently-serving-llms/)
- [Federated learning](../federated-learning/): training that also stays on device
