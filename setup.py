from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='involution_cuda',
    ext_modules=[
        CUDAExtension(
            name='involution_cuda',
            sources=[
                'csrc/involution_kernel_cpp.cpp',
                'csrc/involution_kernel.cu'
            ],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': [
                    '-O2',
                    '--compiler-options', '-fPIC'
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
