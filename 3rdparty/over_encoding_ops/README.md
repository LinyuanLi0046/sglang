# WeLMv4 CUDA host-backed embedding allocator

This optional extension allocates WeLMv4 embedding tables with
`cudaMallocHost` and exposes the mapped UVA pointer as a CUDA tensor. It is
used only when the server starts with `--enable-over-encoding` on CUDA.

From the SGLang repository root, install it into the same environment as
SGLang:

```bash
pip install --no-build-isolation ./3rdparty/over_encoding_ops
```

It requires a CUDA toolkit compatible with the installed PyTorch build. The
flag saves GPU DRAM, but embedding reads cross the host link and can reduce
throughput. Ascend NPU never imports or uses this extension.
