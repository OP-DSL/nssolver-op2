# An alternative to the Makefile: every variant of the OP2 app against an
# installed OP2 (find_package(OP2), OP2_DIR or CMAKE_PREFIX_PATH), which the
# implicit scheme needs built with its solver layer (OP2_ENABLE_PETSC, ...).
#
#   cmake -S . -B build -DOP2_DIR=<prefix>/lib/cmake/OP2 && cmake --build build
#
# The helper tools (preprocessing, VTK conversion) stay with the Makefile.
cmake_minimum_required(VERSION 3.26)
project(nssolver_op2 LANGUAGES C CXX)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()

find_package(OP2 REQUIRED CONFIG)

op2_add_app_variants(
    NAME     nssolver_op2
    LANGUAGE cpp
    SOURCES  nssolver_op2.cpp
    INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include
    EXCLUDE_VARIANTS *_legacy mpi_*_legacy
    WITH_HDF5
    TARGETS_VAR _targets)

# Generated kernels call the helpers beside them in op2_kernels.h, as the
# Makefile's -include gives them.
foreach(_t IN LISTS _targets)
    target_compile_options(${_t} PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX,CUDA>:SHELL:-include ${CMAKE_CURRENT_SOURCE_DIR}/op2_kernels.h>")
endforeach()
