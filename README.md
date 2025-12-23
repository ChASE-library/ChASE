[![License](https://img.shields.io/github/license/ChASE-library/ChASE)](https://github.com/ChASE-library/ChASE/blob/master/LICENSE) [![DOI](https://zenodo.org/badge/349075288.svg)](https://zenodo.org/badge/latestdoi/349075288) [![Latest Version](https://img.shields.io/github/v/release/ChASE-library/ChASE)](https://github.com/ChASE-library/ChASE/releases/latest) [![DOI](https://img.shields.io/badge/DOI-10.1145%2F3313828%20-orange)](https://doi.org/10.1145/3313828) [![DOI](https://img.shields.io/badge/DOI-10.1002%2Fcpe.3394%20-orange)](https://doi.org/10.1002/cpe.3394) [![coverage](https://img.shields.io/endpoint?url=https://gitlab.jsc.fz-juelich.de/wu7/chase-coverage-badges/-/raw/main/coverage/coverage.json)
](https://gitlab.jsc.fz-juelich.de/chase/chase-library/ChASE/-/jobs?kind=BUILD)
 [![JuRSE Code of Month](https://img.shields.io/badge/JuRSE_Code_Pick-Dec_2025-blue?link=https://www.fz-juelich.de/en/rse/community-initiatives/jurse-code-of-the-month/december-2025)](https://www.fz-juelich.de/en/rse/community-initiatives/jurse-code-of-the-month/december-2025)
<img src="docs/images/ChASE_Logo_RGB.png" alt="Matrix Generation Pattern" style="zoom:40%;" />
# ChASE: a Chebyshev Accelerated Subspace Eigensolver for Dense Eigenproblems

The **Ch**ebyshev **A**ccelerated **S**ubspace **E**igensolver (ChASE) is a modern and scalable library based on subspace iteration with polynomial acceleration to solve dense Hermitian (Symmetric) algebraic eigenvalue problems, especially solving dense Hermitian eigenproblems arragend in a sequence. Novel to ChASE is the computation of the spectral estimates that enter in the filter and an optimization of the polynomial degree that further reduces the necessary floating-point operations. 

ChASE is written in C++ using the modern software engineering concepts that favor a simple integration in application codes and a straightforward portability over heterogeneous platforms. When solving sequences of Hermitian eigenproblems for a portion of their extremal spectrum, ChASE greatly benefits from the sequence’s spectral properties and outperforms direct solvers in many scenarios. The library ships with two distinct parallelization schemes, supports execution over distributed GPUs, and is easily extensible to other parallel computing architectures.

## Use Case and Features

- **Real and Complex:** ChASE is templated for real and complex numbers. So it can be used to solve *real symmetric* eigenproblems as well as *complex Hermitian* ones.
- **Eigespectrum:** ChASE algorithm is designed to solve for the *extremal portion* of the eigenspectrum of matrix `A`. The library is particularly efficient when no more than `20%` of the extremal portion of the eigenspectrum is sought after. For larger fractions the subspace iteration algorithm may struggle to be competitive. Converge could become an issue for fractions close to or larger than `50%`.
- **Type of Problem:** ChASE can currently handle only standard eigenvalue problems. 
- **Sequences:** ChASE is particularly efficient when dealing with *sequences of eigenvalue problems*, where the eigenvectors solving for one problem can be use as input to accelerate the solution of the next one.
- **Vectors input:** Since it is based on subspace iteration, ChASE can receive as input a matrix of vector equal to the number of desired eigenvalues. ChASE can experience substantial speed-ups when this input matrix contains some information about the sought after eigenvectors.
- **Degree optimization:** For a fixed accuracy level, ChASE can optimize the degree of the Chebyshev polynomial filter so as to minimize the number of FLOPs necessary to reach convergence.
- **Precision:** ChASE is also templated to work in *Single Precision* (SP) or *Double Precision* (DP).


## Builds of ChASE

ChASE supports different builds for different systems with different architectures:
   - **Shared memory build:** This is the simplest configuration and should be exclusively selected when ChASE is used on only one computing node or on a single GPU. 
   - **MPI+Threads build:** On multi-core homogeneous CPU clusters, ChASE is best used in its pure MPI build. In this configuration, ChASE is typically used with one MPI rank per NUMA domain and as many threads as number of available cores per NUMA domain.
   - **Multi-GPU build:** ChASE can be configured to take advantage of GPUs on heterogeneous computing clusters. Currently we support the use of one GPU per MPI rank. Multiple-GPU per computing node can be used when MPI rank number per node equals to the GPU number per node. 
      - **NCCL Backend:** by default, ChASE uses **[NCCL](https://developer.nvidia.com/nccl)** as backend for the collective communications across different GPUs.
      - **CUDA-Aware MPI Backend**: alternatively, CUDA-Aware MPI can be used for the communications.

## Supported Data types 
   
   ChASE  supports different data types:
   - **Shared memory build** requires dense matrices to be column major.
   - **Distributed-memory build** support two types of data distribution of matrix `A` across 2D MPI/GPU grid:
      - **Block Distribution**:  each MPI rank of 2D grid is assigned a block of dense matrix **A**.
      - **Block-Cyclic Distribution**: an distribution scheme for implementation of dense matrix computations on distributed-memory machines, to improve the load balance of matrix computation if the amount of work differs for different entries of a matrix. For more details, please refer to [Netlib](https://www.netlib.org/scalapack/slug/node75.html) .

## Quick Start

### Installing Dependencies

```bash
#Linux Operating System
sudo apt-get install cmake #install CMake
sudo apt-get install build-essential #install GNU Compiler
sudo apt-get install libopenblas-dev #install BLAS and LAPACK
sudo apt-get install libopenmpi-dev #install MPI

#Apple Mac Operating System 
sudo port install cmake #install CMake
sudo port install gcc10 #install GNU Compiler
sudo port select --set gcc mp-gcc10 #Set installed GCC as C compiler
sudo port install OpenBLAS +native #install BLAS and LAPACK
sudo port install openmpi #install MPI
sudo port select --set mpi openmpi-mp-fortran #Set installed MPI as MPI compiler
```

### Cloning ChASE source code

```bash
git clone https://github.com/ChASE-library/ChASE #cloning the ChASE repository
git checkout v1.0.0 #it is recommended to check out the latest stable tag.
```

### Building and Installing the ChASE library

```bash
cd ChASE/
mkdir build
cd build/
cmake .. -DCMAKE_INSTALL_PREFIX=${ChASEROOT}
make install
```

More details about the installation on both local machine and clusters, please refer to [User Documentation](https://chase-library.github.io/ChASE/quick-start.html) (⚠️**To be updated**).

<!-- a normal html comment 

## Documentation

The documentation of ChASE is available [online](https://chase-library.github.io/ChASE/index.html).

Compiling the documentation in local requires  enable `-DBUILD_WITH_DOCS=ON` flag when compiling ChASE library:

```bash
cmake .. -DBUILD_WITH_DOCS=ON
```
-->
## Examples

Multiple examples are provided, which helps user get familiar with ChASE. 

**Build ChASE with Examples** requires enable `-DCHASE_BUILD_WITH_EXAMPLES=ON` flag when compiling ChASE library:

```bash
cmake .. -DCHASE_BUILD_WITH_EXAMPLES=ON
```

**5 examples are available** in folder [examples](https://github.com/ChASE-library/ChASE/tree/master/examples):

0. The example [0_hello_world](https://github.com/ChASE-library/ChASE/tree/master/examples/0_hello_world) constructs a simple Clement matrix and find a given number of its eigenpairs.

1. The example [1_sequence_eigenproblems](https://github.com/ChASE-library/ChASE/tree/master/examples/1_sequence_eigenproblems) illustrates how ChASE can be used to solve a sequence of eigenproblems. (⚠️**To be included**).
2. The example [2_input_output](https://github.com/ChASE-library/ChASE/tree/master/examples/2_input_output) provides the configuration of parameters of ChASE from command line (supported by Boost); the parallel I/O which loads the local matrices into the computing nodes in parallel.
3. The example [3_installation](https://github.com/ChASE-library/ChASE/tree/master/examples/3_installation) shows the way to link ChASE to other applications.
4. The example [4_interface](https://github.com/ChASE-library/ChASE/tree/master/examples/4_interface) shows examples to use the C and Fortran interfaces of ChASE.

## Developers

### Main developers

- Edoardo Di Napoli – Algorithm design and development
- Xinzhe Wu – Algorithm development, advanced parallel (MPI and GPU) implementation and optimization, developer documentation

### Current contributors

- Clément Richefort - Integration of ChASE into [YAMBO](https://www.yambo-code.eu/) code.
- Davor Davidović – Advanced parallel GPU implementation and optimization
- Nenad Mijić – ARM-based implementation and optimization, CholeskyQR, unitests, parallel IO

### Past contributors

- Xiao Zhang – Integration of ChASE into Jena BSE code
- Miriam Hinzen, Daniel Wortmann – Integration of ChASE into FLEUR code
- Sebastian Achilles – Library benchmarking on parallel platforms, documentation
- Jan Winkelmann – DoS algorithm development and advanced `C++` implementation
- Paul Springer – Advanced GPU implementation
- Marija Kranjcevic – OpenMP `C++` implementation
- Josip Zubrinic – Early GPU algorithm development and implementation
- Jens Rene Suckert – Lanczos algorithm and GPU implementation
- Mario Berljafa – Early `C` and `MPI` implementation using the Elemental library


## Contribution

This Github repository mirrors the principal Gitlab repository hosted at the Juelich Supercomputing Centre. There are two main ways you can contribute: 

1. you can fork the open source ChASE repository on Github (https://github.com/ChASE-library/ChASE). Modify the source code (and relative inlined documentation, if necessary) and then submit a pull request. If you have not contributed to the ChASE library before, we will ask you to agree to a Collaboration Agreement (CLA) before the pull request can be approved. Currentlly there is no automatic mechanism to sign such an agreement and we need you to download the file CLA.pdf (that is part of the repository), print it, sign it, scan it and send it back to chase@fz-juelich.de. Upon reception of your signed CLA, your pull request will be reviewed and then eventually approved.
2. Alternatively, if you want to contribute as a developer stably integrated into this project please contact us at chase@fz-juelich.de with a motivated request of collaboration. We will consider your request and get in touch with you to evaluate if and how to give you access directly to the Gitlab repository where the major developments of this software is carried out.

An automatic process to approve a pull request and sign a CLA is under development and will soon substitute option 1. In the meantime, we ask you for your patience and understanding in having to follow such a time consuming procedure.

## How to Cite the Code

The main reference of ChASE is [1] while [2] provides some early results on scalability and usage on sequences of eigenproblems generated by Materials Science applications. [3] and [5] provides the distributed-memory multi-GPU implementation and performance analysis. 

- [1] J. Winkelmann, P. Springer, and E. Di Napoli. *ChASE: a Chebyshev Accelerated Subspace iteration Eigensolver for sequences of Hermitian eigenvalue problems.* ACM Transaction on Mathematical Software, **45** Num.2, Art.21, (2019). [DOI:10.1145/3313828](https://doi.org/10.1145/3313828) , [[arXiv:1805.10121](https://arxiv.org/abs/1805.10121/) ]
- [2] M. Berljafa, D. Wortmann, and E. Di Napoli. *An Optimized and Scalable Eigensolver for Sequences of Eigenvalue Problems.* Concurrency & Computation: Practice and Experience **27** (2015), pp. 905-922. [DOI:10.1002/cpe.3394](https://onlinelibrary.wiley.com/doi/pdf/10.1002/cpe.3394) , [[arXiv:1404.4161](https://arxiv.org/abs/1404.4161) ].
- [3] X. Wu, D. Davidović, S. Achilles,E. Di Napoli. *ChASE: a distributed hybrid CPU-GPU eigensolver for large-scale hermitian eigenvalue problems.* Proceedings of the Platform for Advanced Scientific Computing Conference (PASC22). [DOI:10.1145/3539781.3539792](https://dl.acm.org/doi/10.1145/3539781.3539792) , [[arXiv:2205.02491](https://arxiv.org/pdf/2205.02491/) ].
- [4] X. Wu, E. Di Napoli. *Advancing the distributed Multi-GPU ChASE library through algorithm optimization and NCCL library.*  Proceedings of the SC'23 Workshops of The International Conference on High Performance Computing, Network, Storage, and Analysis (pp. 1688-1696). [DOI:10.1145/3624062.3624249](https://dl.acm.org/doi/abs/10.1145/3624062.3624249), [[arXiv:2309.15595](https://arxiv.org/pdf/2309.15595)].

## Copyright and License

[3-Clause BSD License (BSD License 2.0)](https://github.com/ChASE-library/ChASE/blob/master/LICENSE)

<!-- @Edo, add CLA here -->

