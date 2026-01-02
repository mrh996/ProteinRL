dir_guard = @mkdir -p $(@D)

FIND := find
CXX := g++
CXXFLAGS += -Wall -O3 -std=c++14 -fPIC  # Ensure C++14 standard is used
LDFLAGS += -lm -lmkl_rt -ltbb

# Make sure CUDA_HOME is correctly set to your CUDA installation path
CUDA_HOME := /usr/local/cuda-12.1
NVCC := $(CUDA_HOME)/bin/nvcc
NVCCFLAGS += --default-stream per-thread -Wno-deprecated-gpu-targets -std=c++17 --use_fast_math --compiler-options '-fPIC'

include_dirs = $(CUDA_HOME)/include \
               /LOCAL2/mur/.conda/envs/PreMut/include \
               /LOCAL2/mur/.conda/envs/PreMut/lib/python3.10/site-packages/torch/include \
               /LOCAL2/mur/.conda/envs/PreMut/lib/python3.10/site-packages/torch/include/torch/csrc/api/include \
               /LOCAL2/mur/.conda/envs/PreMut/lib/python3.10/site-packages/torch/include/TH \
               /LOCAL2/mur/.conda/envs/PreMut/lib/python3.10/site-packages/torch/include/THC

NVCCFLAGS += $(addprefix -I,$(include_dirs))

# Adjust CUDA_ARCH to exclude unsupported architectures
CUDA_ARCH := -gencode arch=compute_50,code=sm_50 \
             -gencode arch=compute_60,code=sm_60 \
             -gencode arch=compute_70,code=sm_70 \
             -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=sm_86  # Add this if your hardware supports it

build_root = _ext
obj_build_root = $(build_root)

cu_files = $(shell $(FIND) src/ -name "*.cu" -printf "%P\n")
cu_obj_files = $(subst .cu,.o,$(cu_files))
objs = $(addprefix $(obj_build_root)/,$(cu_obj_files))

DEPS = ${objs:.o=.d}
mylib = _ext/my_lib/_my_lib.so

all: $(objs) $(mylib)

$(obj_build_root)/%.o: src/%.cu
	$(dir_guard)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -M $< -o ${@:.o=.d} -odir $(@D)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

$(mylib): src/*.c src/*.h src/*.cu
	python build.py build_ext --inplace

clean:
	rm -f $(obj_build_root)/*.o
	rm -f $(obj_build_root)/*.d
	rm -rf _ext
	rm -f functions/*.pyc
	rm -f modules/*.pyc

-include $(DEPS)
