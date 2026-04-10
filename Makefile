OP2_COMMON ?= /home/ireguly/OP2-Common
OP2_INSTALL_PATH ?= $(OP2_COMMON)/op2
H5CXX ?= h5c++

include $(OP2_COMMON)/makefiles/common.mk

APP_NAME := nssolver_op2
APP_SRC := nssolver_op2.cpp
APP_INC := -I. -Iinclude
OP2_LIBS_WITH_HDF5 := true

CXXFLAGS += -std=c++20 -include op2_kernels.h
NVCCFLAGS += -include op2_kernels.h

HELPER_DIR := .helpers
HELPER_BINDIR := $(HELPER_DIR)/bin
HELPER_OBJDIR := $(HELPER_DIR)/obj
HELPER_CXXFLAGS := -std=c++20 -Wall -Wextra -Wpedantic -DNSSOLVER_HAVE_HDF5=1 -Iinclude
HELPER_COMMON_SRCS := \
	src/mesh.cpp \
	src/physics.cpp \
	src/flux.cpp \
	src/solver.cpp \
	src/validation.cpp \
	src/vtk_writer.cpp \
	src/hydra_reader.cpp \
	src/hydra_benchmark.cpp
HELPER_COMMON_OBJS := $(patsubst src/%.cpp,$(HELPER_OBJDIR)/%.o,$(HELPER_COMMON_SRCS))

include $(OP2_COMMON)/makefiles/c_app.mk

.PHONY: seq genseq openmp helpers-config helpers-build helper-tools preprocess-box preprocess-bump preprocess-flatplate preprocess-hydra smoke consistency flatplate

seq: $(APP_NAME)_seq
genseq: $(APP_NAME)_genseq
openmp: $(APP_NAME)_openmp

helpers-config:
	./scripts/configure_helpers.sh

helpers-build: helper-tools

helper-tools: $(HELPER_BINDIR)/nssolver_demo_local $(HELPER_BINDIR)/nssolver_preprocess_op2_helper $(HELPER_BINDIR)/nssolver_hdf5_to_vtk_helper $(HELPER_BINDIR)/nssolver_op2_benchmark_postprocess_helper

preprocess-box: helpers-build
	./scripts/preprocess_mesh.sh box meshes-op2/box.h5

preprocess-bump: helpers-build
	./scripts/preprocess_mesh.sh bump meshes-op2/bump.h5

preprocess-flatplate: helpers-build
	./scripts/preprocess_mesh.sh flatplate meshes-op2/flatplate.h5

preprocess-hydra: helpers-build
	./scripts/preprocess_mesh.sh hydra meshes-op2/hydra.h5 meshes/hydra.jm70.grid.1.hdf

smoke:
	./tests/test_smoke.sh

consistency:
	./tests/test_consistency.sh

flatplate:
	./scripts/run_flatplate_validation.sh

$(HELPER_OBJDIR) $(HELPER_BINDIR):
	mkdir -p $@

$(HELPER_OBJDIR)/%.o: src/%.cpp | $(HELPER_OBJDIR)
	$(H5CXX) $(HELPER_CXXFLAGS) -c $< -o $@

$(HELPER_BINDIR)/nssolver_demo_local: $(HELPER_COMMON_OBJS) $(HELPER_OBJDIR)/main.o | $(HELPER_BINDIR)
	$(H5CXX) $(HELPER_CXXFLAGS) $^ -o $@

$(HELPER_BINDIR)/nssolver_preprocess_op2_helper: $(HELPER_COMMON_OBJS) $(HELPER_OBJDIR)/preprocess_op2.o | $(HELPER_BINDIR)
	$(H5CXX) $(HELPER_CXXFLAGS) $^ -o $@

$(HELPER_BINDIR)/nssolver_hdf5_to_vtk_helper: $(HELPER_COMMON_OBJS) $(HELPER_OBJDIR)/hdf5_to_vtk.o | $(HELPER_BINDIR)
	$(H5CXX) $(HELPER_CXXFLAGS) $^ -o $@

$(HELPER_BINDIR)/nssolver_op2_benchmark_postprocess_helper: $(HELPER_COMMON_OBJS) $(HELPER_OBJDIR)/op2_benchmark_postprocess.o | $(HELPER_BINDIR)
	$(H5CXX) $(HELPER_CXXFLAGS) $^ -o $@
