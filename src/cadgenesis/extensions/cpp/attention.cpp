/**
 * C++ extension for CADGenesis-LM CUDA attention operations.
 * 
 * Provides Python-callable functions that wrap CUDA kernels
 * for accelerated attention computation in the GeometryAwareTransformer.
 */

#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include "attention.cu.h" // Include generated header from CUDA kernel

// Forward declaration of the CUDA kernel function
at::Tensor cuda_attention_forward(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor mask,
    double scale);

at::Tensor cuda_attention_backward(
    at::Tensor grad_output,
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor mask,
    double scale) {
  auto result = cuda_attention_backward(
      grad_output, q, k, v, mask, scale);
  return std::get<0>(result); // Simplified for example
}

// PyBind11 wrapper for forward attention
at::Tensor attention_forward_cuda(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor mask, double scale) {
  TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
  TORCH_CHECK(k.is_cuda(), "k must be on CUDA");
  TORCH_CHECK(v.is_cuda(), "v must be on CUDA");
  TORCH_CHECK(mask.is_cuda(), "mask must be on CUDA");
  
  return cuda_attention_forward(q, k, v, mask, scale);
}

// PyBind11 wrapper for backward attention
std::tuple<at::Tensor, at::Tensor, at::Tensor> attention_backward_cuda(
    at::Tensor grad_output,
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor mask,
    double scale) {
  TORCH_CHECK(grad_output.is_cuda(), "grad_output must be on CUDA");
  TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
  TORCH_CHECK(k.is_cuda(), "k must be on CUDA");
  TORCH_CHECK(v.is_cuda(), "v must be on CUDA");
  TORCH_CHECK(mask.is_cuda(), "mask must be on CUDA");
  
  auto [grad_q, grad_k, grad_v] = cuda_attention_backward(grad_output, q, k, v, mask, scale);
  return std::make_tuple(grad_q, grad_k, grad_v);
}

PYBIND11_EXTENSION_MODULE(cadgenesis_cpp_ext, m) {
  m.doc() = "C++ extension for CADGenesis-LM CUDA operations";
  
  m.def("attention_forward_cuda", &attention_forward_cuda,
        "Forward pass of CUDA-accelerated attention",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("mask"), py::arg("scale"));
  
  m.def("attention_backward_cuda", &attention_backward_cuda,
        "Backward pass of CUDA-accelerated attention",
        py::arg("grad_output"), py::arg("q"), py::arg("k"), py::arg("v"), py::arg("mask"), py::arg("scale"));
}