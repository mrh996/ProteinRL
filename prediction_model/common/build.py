import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

this_file = os.path.dirname(os.path.realpath(__file__))

sources = ['src/my_lib.c']
headers = ['src/my_lib.h']
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/my_lib_cuda.cu']  # Ensure the CUDA source file has the correct extension
    headers += ['src/my_lib_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

extra_objects = ['_ext/custom_kernel.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ext_modules = []

if with_cuda:
    ext_modules.append(
        CUDAExtension(
            name='_ext.my_lib',
            sources=sources,
            include_dirs=['src'],
            define_macros=defines,
            extra_objects=extra_objects,
            extra_compile_args={
                'cxx': ['-fopenmp'],
                'nvcc': ['-O2', '-arch=sm_75']  # Adjust the architecture as needed
            },
            extra_link_args=['-lgomp']
        )
    )
else:
    ext_modules.append(
        CppExtension(
            name='_ext.my_lib',
            sources=sources,
            include_dirs=['src'],
            define_macros=defines,
            extra_objects=extra_objects,
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-lgomp']
        )
    )

setup(
    name='my_lib',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    }
)
