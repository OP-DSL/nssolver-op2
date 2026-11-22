#!/bin/bash -l
#SBATCH -p gpu --gres=gpu:1 -c 32 -t 04:00:00 -J op2-libs
#SBATCH -o /scratch/c_ppkeitk/fv2k/op2sol/logs/build-%j.out
# Builds the solver stack for OP2 and nssolver on a GPU node (sm_80, Milan):
#   STEPS="venv hdf5 metis hypre petsc ginkgo op2 nssolver"  (default: all)
# Each step skips itself when its install is there; rm the install to redo it.
source /scratch/c_ppkeitk/fv2k/op2sol/scripts/env.sh
set -u
J=${J:-32}
SRC=$ROOT/src; mkdir -p $SRC $PREFIX $ROOT/build $ROOT/logs
[ -e $PREFIX/lib ] || ln -s lib64 $PREFIX/lib   # RHEL installs go to lib64
STEPS=${STEPS:-"venv hdf5 metis hypre petsc ginkgo op2 nssolver"}
fetch() { # url dir: unpack, dropping a wrapper directory if the archive has two levels
  [ -d $SRC/$2 ] && return
  mkdir -p $SRC/$2 && curl -sL $1 | tar xz -C $SRC/$2 --strip-components=1 || return 1
  if [ $(ls $SRC/$2 | wc -l) -eq 1 ] && [ -d $SRC/$2/* ]; then mv $SRC/$2/*/* $SRC/$2/; fi
}
CUDAARCH=80
for s in $STEPS; do
  echo "== $s $(date -Is)"
  case $s in
  venv)
    [ -x $ROOT/venv/bin/python ] || { /opt/software/packages/miniconda3/bin/python3 -m venv $ROOT/venv &&
      $ROOT/venv/bin/pip install -q -r $ROOT/src/op2/translator-v2/requirements.txt pkgconf; } ;;
  hdf5)
    [ -f $PREFIX/hdf5/lib/cmake/hdf5/hdf5-config.cmake ] || {
      fetch https://github.com/HDFGroup/hdf5/releases/download/hdf5_1.14.6/hdf5-1.14.6.tar.gz hdf5
      cmake -S $SRC/hdf5 -B $ROOT/build/hdf5 -GNinja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=$PREFIX/hdf5 -DHDF5_ENABLE_PARALLEL=ON -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTING=OFF -DHDF5_BUILD_EXAMPLES=OFF -DHDF5_BUILD_TOOLS=ON \
        -DMPI_C_COMPILER=$CC -DHDF5_ENABLE_Z_LIB_SUPPORT=OFF &&
      cmake --build $ROOT/build/hdf5 -j$J && cmake --install $ROOT/build/hdf5; } ;;
  metis) # 64-bit indices: OP2's idx_g_t is long long
    [ -f $PREFIX/lib/libparmetis.a ] || {
      fetch https://github.com/KarypisLab/GKlib/archive/refs/heads/master.tar.gz gklib
      fetch https://github.com/KarypisLab/METIS/archive/refs/heads/master.tar.gz metis
      fetch https://github.com/KarypisLab/ParMETIS/archive/refs/heads/main.tar.gz parmetis
      (cd $SRC/gklib && make config prefix=$PREFIX cc=gcc >/dev/null && make -j$J install >/dev/null) &&
      (cd $SRC/metis && make config prefix=$PREFIX gklib_path=$PREFIX cc=$CC i64=1 >/dev/null && make -j$J install >/dev/null) &&
      (cd $SRC/parmetis && make config prefix=$PREFIX gklib_path=$PREFIX metis_path=$PREFIX \
         cc=$CC cxx=$CXX >/dev/null && make -j$J install >/dev/null); }
    ls $PREFIX/lib/libparmetis.a $PREFIX/lib/libmetis.a ;;
  hypre)
    [ -d $PREFIX/hypre/lib64/cmake/HYPRE ] || {
      fetch https://github.com/hypre-space/hypre/archive/refs/tags/v3.2.1.tar.gz hypre
      cmake -S $SRC/hypre/src -B $ROOT/build/hypre -GNinja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=$PREFIX/hypre -DHYPRE_ENABLE_MIXEDINT=ON -DHYPRE_ENABLE_CUDA=ON \
        -DHYPRE_ENABLE_UMPIRE=OFF -DCMAKE_CUDA_ARCHITECTURES=$CUDAARCH -DHYPRE_WITH_MPI=ON \
        -DMPI_C_COMPILER=$CC -DMPI_CXX_COMPILER=$CXX \
        -DCMAKE_CUDA_FLAGS=-I$MPICH_DIR/include -DBUILD_SHARED_LIBS=ON &&
      cmake --build $ROOT/build/hypre -j$J && cmake --install $ROOT/build/hypre; } ;;
  petsc)
    [ -f $PREFIX/petsc/lib/libpetsc.so ] || {
      fetch https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-3.26.1.tar.gz petsc
      (cd $SRC/petsc && unset PETSC_ARCH && PETSC_DIR=$SRC/petsc ./configure --prefix=$PREFIX/petsc \
        --with-cc=$CC --with-cxx=$CXX --with-mpi=1 --with-fc=0 --with-debugging=0 --with-shared-libraries=1 \
        COPTFLAGS=-O3 CXXOPTFLAGS=-O3 CUDAOPTFLAGS=-O3 --with-cuda=1 --with-cuda-dir=$CUDA_HOME \
        --with-cuda-arch=$CUDAARCH --download-f2cblaslapack --with-python=0 &&
       make PETSC_DIR=$SRC/petsc PETSC_ARCH=arch-linux-c-opt -j$J all && make PETSC_DIR=$SRC/petsc PETSC_ARCH=arch-linux-c-opt install); } ;;
  ginkgo)
    [ -d $PREFIX/ginkgo ] || {
      fetch https://github.com/ginkgo-project/ginkgo/archive/refs/tags/v1.10.0.tar.gz ginkgo
      cmake -S $SRC/ginkgo -B $ROOT/build/ginkgo -GNinja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=$PREFIX/ginkgo -DGINKGO_BUILD_CUDA=ON -DGINKGO_BUILD_OMP=ON \
        -DGINKGO_BUILD_MPI=ON -DGINKGO_BUILD_TESTS=OFF -DGINKGO_BUILD_EXAMPLES=OFF \
        -DGINKGO_BUILD_BENCHMARKS=OFF -DGINKGO_BUILD_HIP=OFF -DGINKGO_BUILD_SYCL=OFF \
        -DCMAKE_CUDA_ARCHITECTURES=$CUDAARCH -DMPI_C_COMPILER=$CC \
        -DMPI_CXX_COMPILER=$CXX &&
      cmake --build $ROOT/build/ginkgo -j$J && cmake --install $ROOT/build/ginkgo; } ;;
  op2)
    PY=$ROOT/venv/bin/python
    cmake -S $SRC/op2 -B $ROOT/build/op2 -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$OP2_PREFIX -DBUILD_SHARED_LIBS=ON -DCMAKE_CUDA_ARCHITECTURES=$CUDAARCH \
      -DOP2_ENABLE_CUDA=ON -DOP2_ENABLE_MPI=ON -DOP2_ENABLE_OPENMP=ON -DOP2_ENABLE_HDF5=ON \
      -DOP2_ENABLE_HIP=OFF -DOP2_ENABLE_FORTRAN=OFF -DOP2_ENABLE_AMGX=OFF -DOP2_ENABLE_ROCALUTION=OFF \
      -DOP2_ENABLE_PETSC=ON -DOP2_ENABLE_HYPRE=ON -DOP2_ENABLE_GINKGO=ON -DOP2_ENABLE_SOLVE_MOCK=ON \
      -DOP2_ENABLE_TESTS=ON -DOP2_ENABLE_APPS=ON -DOP2_ENABLE_TRANSLATOR=ON -DOP2_ENABLE_INSTALL=ON \
      "-DOP2_IMPLICIT_NV=1;4;6" -DPython3_EXECUTABLE=$PY -DPKG_CONFIG_EXECUTABLE=$ROOT/venv/bin/pkg-config \
      -DMPI_C_COMPILER=$CC -DMPI_CXX_COMPILER=$CXX -DMPI_C_ADDITIONAL_INCLUDE_DIRS=$MPICH_DIR/include -DMPI_CXX_ADDITIONAL_INCLUDE_DIRS=$MPICH_DIR/include -DHDF5_ROOT=$PREFIX/hdf5 -DParMETIS_ROOT=$PREFIX -DMETIS_ROOT=$PREFIX \
      -DHYPRE_DIR=$PREFIX/hypre/lib64/cmake/HYPRE -DPETSC_DIR=$PREFIX/petsc -DPETSC_ARCH= \
      -DGinkgo_DIR=$PREFIX/ginkgo/lib/cmake/Ginkgo &&
    cmake --build $ROOT/build/op2 -j$J -- -k 0 ; cmake --install $ROOT/build/op2 ;;
  nssolver)
    cmake -S $SRC/nssolver -B $ROOT/build/nssolver -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=$CUDAARCH -DMPI_C_COMPILER=$CC -DMPI_CXX_COMPILER=$CXX -DMPI_C_ADDITIONAL_INCLUDE_DIRS=$MPICH_DIR/include -DMPI_CXX_ADDITIONAL_INCLUDE_DIRS=$MPICH_DIR/include -DOP2_DIR=$OP2_PREFIX/lib64/cmake/op2 -DPKG_CONFIG_EXECUTABLE=$ROOT/venv/bin/pkg-config \
      -DHDF5_ROOT=$PREFIX/hdf5 -DHYPRE_DIR=$PREFIX/hypre/lib64/cmake/HYPRE -DGinkgo_DIR=$PREFIX/ginkgo/lib64/cmake/Ginkgo -DPETSC_DIR=$PREFIX/petsc -DPETSC_ARCH= \
      -DPython3_EXECUTABLE=$ROOT/venv/bin/python &&
    cmake --build $ROOT/build/nssolver -j$J -- -k 0 ;;
  esac
done
echo "== done $(date -Is)"
