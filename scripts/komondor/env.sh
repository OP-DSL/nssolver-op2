# OP2 and nssolver on komondor (HPE Cray EX): source on a login or compute node.
#   ROOT   everything lives here (sources, builds, installs, meshes, runs)
# Cray MPICH through the cc/CC wrappers (GPU-aware via craype-accel-nvidia80, so
# set MPICH_GPU_SUPPORT_ENABLED=1 for device buffers), PrgEnv-gnu, CUDA 12.9.
export ROOT=${ROOT:-/scratch/c_ppkeitk/fv2k/op2sol}
export PREFIX=$ROOT/install
module load PrgEnv-gnu >/dev/null 2>&1
module load gcc-native/13.2 cuda/12.9 craype-accel-nvidia80 cray-mpich cmake/3.30.8 >/dev/null 2>&1
# Intra-node single-copy transfers (XPMEM, CMA) abort 4-rank runs in MPI_Allgather
# ("Invalid count", lmt_import_mem) on the gpu nodes; NONE works.
export MPICH_SMP_SINGLE_COPY_MODE=NONE
export CRAYPE_LINK_TYPE=dynamic   # the wrappers link statically by default
# libclang (the translator) cannot see the wrappers' implicit -I: CPATH it.
export CPATH=$MPICH_DIR/include${CPATH:+:$CPATH}
export CUDA_HOME=/opt/software/packages/cuda/12.9
export MPI_HOME=$MPICH_DIR
# Absolute paths: CMake searches /usr/bin before PATH, and would find the system cc.
export CC=$(command -v cc) CXX=$(command -v CC) CUDAHOSTCXX=g++
export PATH=$ROOT/venv/bin:$PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$PREFIX/hdf5/lib:$PREFIX/lib:$PREFIX/lib64:${LD_LIBRARY_PATH:-}
export HDF5_ROOT=$PREFIX/hdf5 HYPRE_ROOT=$PREFIX/hypre PETSC_DIR=$PREFIX/petsc PETSC_ARCH=
export OP2_PREFIX=$PREFIX/op2
