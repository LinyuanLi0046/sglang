from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="over_encoding_ops",
    version="0.1.0",
    packages=["over_encoding_ops"],
    ext_modules=[
        CUDAExtension(
            name="over_encoding_ops_kernel",
            sources=["csrc/host_allocator.cpp"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--expt-relaxed-constexpr"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

