#!/bin/bash
# Sources rsynced to ~/op2sol/src/{op2,nssolver} (OP2 from the solvers branch), PETSc tarball in ~/op2sol/src.
# PETSc (CUDA), OP2 (solver layer, PETSc), nssolver and its helpers on renyi (A100).
#SBATCH -p gpu -w renyi -N1 -c32 --gres=gpu:1 -t 04:00:00 -J op2sol-build
#SBATCH -o /home/reguly/op2sol/logs/build-%j.out
module load OP2-env gnu12/12.3.0 2>&1
# ARCH (default 80, renyi's A100s) picks the CUDA architecture; another one
# builds into install$ARCH and build$ARCH (70 for the V100 nodes:
# ARCH=70 sbatch -w neumann scripts/cl/build.sbatch).
ARCH=${ARCH:-80}
SUFFIX=$([ "$ARCH" = 80 ] || echo "$ARCH")
R=$HOME/op2sol
I=$R/install$SUFFIX
B=$R/build$SUFFIX
mkdir -p $B
set -e
echo "gcc: $(gcc --version | head -1); nvcc: $(nvcc --version | tail -1); mpicc: $(mpicc --version | head -1)"

if [ ! -f $I/petsc/lib/libpetsc.so ]; then
  cd $R/src
  [ -d petsc-3.26.1$SUFFIX ] || { mkdir petsc-3.26.1$SUFFIX && tar xzf petsc-3.26.1.tar.gz -C petsc-3.26.1$SUFFIX --strip-components=1; }
  cd petsc-3.26.1$SUFFIX
  unset PETSC_ARCH PETSC_DIR
  ./configure --prefix=$I/petsc --with-mpi-dir=$MPI_INSTALL_PATH --with-fc=0 --with-debugging=0 \
    --with-shared-libraries=1 --download-f2cblaslapack --with-cuda=1 --with-cuda-arch=$ARCH \
    COPTFLAGS=-O2 CXXOPTFLAGS=-O2 CUDAOPTFLAGS=-O2 > $B/petsc-configure.log 2>&1 || { tail -40 $B/petsc-configure.log; exit 1; }
  make -j32 > $B/petsc-make.log 2>&1 || { tail -40 $B/petsc-make.log; exit 1; }
  make install > $B/petsc-install.log 2>&1
fi
echo "petsc ok"

if [ ! -f $I/op2/lib/cmake/op2/OP2Config.cmake ] || [ -n "$REBUILD_OP2" ]; then
  cmake -S $R/src/op2 -B $B/op2 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$I/op2 \
    -DBUILD_SHARED_LIBS=ON -DOP2_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$ARCH -DOP2_ENABLE_MPI=ON \
    -DOP2_ENABLE_OPENMP=ON -DOP2_ENABLE_HDF5=ON -DHDF5_ROOT=$HOME/hdf5-cmake-par \
    -DOP2_ENABLE_PETSC=ON -DPETSC_DIR=$I/petsc -DOP2_ENABLE_HYPRE=OFF -DOP2_ENABLE_FORTRAN=OFF \
    -DOP2_ENABLE_TESTS=OFF -DOP2_ENABLE_APPS=ON -DOP2_ENABLE_INSTALL=ON \
    -DCMAKE_C_FLAGS="-I$MPI_INSTALL_PATH/include" -DCMAKE_CXX_FLAGS="-I$MPI_INSTALL_PATH/include" \
    > $B/op2-cmake.log 2>&1 || { tail -40 $B/op2-cmake.log; exit 1; }
  grep -E "Translator|HDF5|MPI|PETSc|CUDA" $B/op2-cmake.log | head -20
  cmake --build $B/op2 -j32 > $B/op2-make.log 2>&1 || { grep -m20 -E "error" $B/op2-make.log; exit 1; }
  cmake --install $B/op2 > $B/op2-install.log 2>&1
fi
echo "op2 ok"

OP2CFG=$(dirname $(find $I/op2 -name OP2Config.cmake | head -1))
cmake -S $R/src/nssolver -B $B/ns -DOP2_DIR=$OP2CFG -DCMAKE_CUDA_ARCHITECTURES=$ARCH \
  -DCMAKE_C_FLAGS="-I$MPI_INSTALL_PATH/include" -DCMAKE_CXX_FLAGS="-I$MPI_INSTALL_PATH/include" \
  > $B/ns-cmake.log 2>&1 || { tail -40 $B/ns-cmake.log; exit 1; }
cmake --build $B/ns -j32 --target nssolver_op2_cuda nssolver_op2_openmp nssolver_op2_mpi_cuda \
  > $B/ns-make.log 2>&1 || { grep -m20 -E "error" $B/ns-make.log; exit 1; }
ls -la $B/ns/nssolver_op2_*

cd $R/src/nssolver && mkdir -p .helpers/bin
H=$HOME/hdf5-cmake-par
COMMON="src/mesh.cpp src/physics.cpp src/validation.cpp src/vtk_writer.cpp src/hydra_reader.cpp src/su2_reader.cpp src/hydra_benchmark.cpp"
mpicxx -O2 -std=c++20 -DNSSOLVER_HAVE_HDF5=1 -Iinclude -I$H/include $COMMON src/preprocess_op2.cpp \
  -L$H/lib -Wl,-rpath,$H/lib -lhdf5 -ldl -lm -o .helpers/bin/nssolver_preprocess_op2_helper
mpicxx -O2 -std=c++20 -DNSSOLVER_HAVE_HDF5=1 -Iinclude -I$H/include $COMMON src/op2_benchmark_postprocess.cpp \
  -L$H/lib -Wl,-rpath,$H/lib -lhdf5 -ldl -lm -o .helpers/bin/nssolver_op2_benchmark_postprocess_helper
ls -la .helpers/bin
echo BUILD-DONE
