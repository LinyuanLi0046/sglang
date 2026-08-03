#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
  TORCH_CHECK(
      status == cudaSuccess,
      operation,
      " failed: ",
      cudaGetErrorString(status));
}

}  // namespace

// Allocate page-locked host memory and expose its UVA address as a CUDA tensor.
// CUDA kernels can read this memory directly, while the table does not consume
// ordinary device DRAM.  The trade-off is host-link bandwidth/latency.
torch::Tensor host_mapped_empty(
    at::IntArrayRef sizes,
    torch::ScalarType dtype,
    int64_t device_id) {
  int64_t numel = 1;
  for (const auto size : sizes) {
    TORCH_CHECK(size >= 0, "negative tensor dimension: ", size);
    numel *= size;
  }

  check_cuda(cudaSetDevice(static_cast<int>(device_id)), "cudaSetDevice");
  const size_t bytes = static_cast<size_t>(numel) * torch::elementSize(dtype);
  void* host_ptr = nullptr;
  check_cuda(cudaMallocHost(&host_ptr, bytes), "cudaMallocHost");

  auto deleter = [](void* ptr) {
    if (ptr != nullptr) {
      // A deleter cannot propagate an exception.  Allocation failures are
      // checked above; release errors are intentionally ignored here.
      cudaFreeHost(ptr);
    }
  };
  return torch::from_blob(
      host_ptr,
      sizes,
      deleter,
      torch::TensorOptions().dtype(dtype).device(
          torch::kCUDA, static_cast<int>(device_id)));
}

PYBIND11_MODULE(over_encoding_ops_kernel, module) {
  module.doc() = "CUDA mapped-host allocator for WeLMv4 embedding tables";
  module.def(
      "custom_empty",
      &host_mapped_empty,
      pybind11::arg("sizes"),
      pybind11::arg("dtype") = torch::kFloat32,
      pybind11::arg("device_id") = 0);
}

