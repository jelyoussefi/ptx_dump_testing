#----------------------------------------------------------------------------------------------------------------------
# Flags
#----------------------------------------------------------------------------------------------------------------------
SHELL:=/bin/bash

CURRENT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUILD_DIR = ${CURRENT_DIR}/build
ONEAPI_ROOT ?= /opt/intel/oneapi


export TERM=xterm

CXX_COMPILER=clang++
CXX_FLAGS="  -fsycl  -fsycl-targets=nvptx64-nvidia-cuda -Wno-deprecated-declarations  "

#----------------------------------------------------------------------------------------------------------------------
# Targets
#----------------------------------------------------------------------------------------------------------------------
default: run 
.PHONY: build 


build_device:
	@$(call msg,Building for device only   ...)
	@${CXX_COMPILER} -fsycl-device-only -fno-sycl-use-bitcode  -stdlib=libstdc++   ./nvidia_mat_mul.cpp
	

build_host:
	@$(call msg,Building for host only   ...)
	@${CXX_COMPILER} --cuda-compile-host-device -fno-sycl-use-bitcode  -stdlib=libstdc++   ./nvidia_mat_mul.cpp	
	
build:  
	@$(call msg,Building multiplication Application   ...)
	@mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR} && \
		bash -c  ' \
			  CXX=${CXX_COMPILER} \
			  CXXFLAGS=${CXX_FLAGS} \
			  cmake -B . -S .. && \
			  make '

run: build
	@$(call msg,Running the mat multiplication Application ...)
	@${BUILD_DIR}/matrix_ops
		

clean:
	@rm -rf ${BUILD_DIR} *.spv
