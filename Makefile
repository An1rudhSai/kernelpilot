PYTHON ?= python
BUILD_DIR ?= build
JOBS ?= 8

# rocm or cuda, taken from the installed PyTorch build.
BACKEND ?= $(shell $(PYTHON) -c "import torch; print('rocm' if torch.version.hip else 'cuda')")
TORCH_CMAKE := $(shell $(PYTHON) -c "import torch; print(torch.utils.cmake_prefix_path)")

CMAKE_ARGS ?=
ifeq ($(BACKEND),rocm)
  ROCM_PATH ?= /opt/rocm
  export HIPCXX ?= $(ROCM_PATH)/llvm/bin/clang++
  HIP_ARCH ?= $(shell $(ROCM_PATH)/bin/rocm_agent_enumerator 2>/dev/null | grep -v gfx000 | head -1)
  ifneq ($(HIP_ARCH),)
    CMAKE_ARGS += -DCMAKE_HIP_ARCHITECTURES=$(HIP_ARCH)
  endif
  CMAKE_ARGS += -DCMAKE_PREFIX_PATH="$(TORCH_CMAKE);$(ROCM_PATH)"
else
  ifdef CUDA_ARCH
    CMAKE_ARGS += -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH)
  endif
  CMAKE_ARGS += -DCMAKE_PREFIX_PATH="$(TORCH_CMAKE)"
endif

.PHONY: build test test-gpu test-cpu bench roofline evaluate lint format clean

build:
	cmake -S . -B $(BUILD_DIR) -DKP_BACKEND=$(BACKEND) -DCMAKE_BUILD_TYPE=Release $(CMAKE_ARGS)
	cmake --build $(BUILD_DIR) -j $(JOBS)

# All correctness checks (GPU kernels + CPU agent logic).
test: build
	$(PYTHON) -m pytest tests -q

test-gpu: build
	$(PYTHON) -m pytest tests/gpu -q

test-cpu:
	$(PYTHON) -m pytest tests/cpu -q

bench: build
	$(PYTHON) -m kernelpilot.cli bench --out results

roofline:
	$(PYTHON) -m kernelpilot.cli roofline --results results/bench.json --out results/roofline.png

evaluate:
	$(PYTHON) -m kernelpilot.cli evaluate --results results/bench.json --runs runs --out results/evaluation.md

lint:
	ruff check .

format:
	ruff format .

clean:
	rm -rf $(BUILD_DIR)
