# *ng*AP: Non-blocking Large-scale Automata Processing on GPUs

This is the repository for ASPLOS'24 paper: ngAP: Non-blocking Large-scale Automata Processing on GPUs.

ngAP is a GPU-based automata processing engine that allows concurrent processing of multiple symbols and enables a broader range of optimizations.

<!-- - [Paper, Slides, Talk] -->

## 0. Requirements

- Hardware:

    ```
    - CPU x86_64 with host memory >= 32GB
    - NVIDIA GPU (arch>=sm_50) with devcie memory >= 24GB
    ```

     We have tested our project on an NVIDIA RTX 3090 (Ampere architecure, 24 GB memory) and an NVIDIA Tesla V100 SXM2 (Volta architecture, 32 GB memory).

- OS & Software:

    ``` bash
    - Ubuntu 20.04
    - GCC 9.4.0
    - GCC 5.3.1, boost 1.71, Ragel, nasm, sqlite3   # for Hyperscan
    - CMake >= 3.24.1
    - CUDA >= 12.0 and NVCC >= 12.0
    - TBB 2020.1                                    # for validation
    - Python >= 3.8
    - numpy scipy pandas seaborn adjustText         # python packages for plotting
    ```

## 1. Environment Setup

### 1.1 Download the Repository and Benchmark

```bash
git clone --recursive git@github.com:getianao/ngAP.git
cd ngAP && source env.sh && echo ${NGAP_ROOT}       # set environment variables

# Download benchmarks: 2.5G
wget https://hkustgz-my.sharepoint.com/:u:/g/personal/tge601_connect_hkust-gz_edu_cn/EbRBcgYV7Z1KrGLk56PjswsBAmdDwfen2zdXTknP5owEAg\?e\=5bWc4W\&download=1 -O automata_benchmark_original.tar.gz
tar -zxvf automata_benchmark_original.tar.gz
```

### 1.2 Install Dependencies

We recommend to use Docker to setup the environment. We provide a [dockerfile](docker/Dockerfile) in the docker folder. 
You can also setup the environment manually.

#### 1.2.1 Docker (Recommended)

If you don't have Docker installed, please follow the [NVIDIA Container Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) to install Docker using the following commands:

```bash
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
  && \
    sudo apt-get update

sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

Once Docker is installed, run the following commands to build the docker image and run the container.
These commands will take approximately 30 minutes to complete:

```bash
docker build -t ngap-ae ${NGAP_ROOT}/docker
docker run -it --rm --gpus all -v ${NGAP_ROOT}:/ngAP ngap-ae:latest /bin/bash
```

After running these commands, you will find yourself inside the container's bash shell.

#### 1.2.2 Manual Setup
Install system packages:
```bash
sudo apt-get install -y libtbb-dev=2020.1-2 cmake
sudo apt-get install -y ragel libboost-all-dev nasm libsqlite3-dev pkg-config g++-5 gcc-5 # Hyperscan
```

If you use `conda` and `pip`, simply run the following commands to install plotting packages:

```bash
conda install -y numpy scipy pandas seaborn -c conda-forge
pip install https://github.com/getianao/figurePlotter/archive/refs/tags/v0.23.9.14.tar.gz
```


## 2 Installation

### 2.1 GPU Schemes
To build GPU executables, run the following commands:

```bash
cd ${NGAP_ROOT}/code && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

The GPU executables will be located in the `${NGAP_ROOT}/code/build/bin` folder, including:

- `ppopp12`: NFA-CG (PPoPP 2012)
- `asyncap`: AsyncAP (SIGMETRICS 2023)
- `obat`: GPU-NFA (ASPLOS 2020)
- `ngap`: ngAP (Our design)

### 2.2 CPU Schemes

To build Hyperscan, run the following commands:

```bash
cd ${NGAP_ROOT}/hscompile/lib/hyperscan && mkdir -p build && cd build
cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc .. 
make -j
cd ${NGAP_ROOT}/hscompile/lib/mnrl/C++
sed -i 's/CC = .*/CC = g++-5/g' Makefile     # requires GCC-5.
make                                         # If an error occurs, try to run it again  
cd ${NGAP_ROOT}/hscompile && mkdir -p build && cd build
cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc       \
    -DHS_SOURCE_DIR=${NGAP_ROOT}/hscompile/lib/hyperscan    \
    -DMNRL_SOURCE_DIR=${NGAP_ROOT}/hscompile/lib/mnrl/C++   \
    ..
make -j
```

The CPU executables will be located in the `${NGAP_ROOT}/hscompile/build` folder, including:

- `hsrun`: Hyperscan (NSDI 2019)

### 2.3 Basic Usage of ngAP

```bash
ngap -a [anml_file] -i [input_stream_file] --algorithm [algorithm] ...
```

The algorithm could be:

- `blockinggroups` : a baseline blocking automata processing (BAP)
- `NAPgroups` : the non-blocking automata processing (ngAP)
- `nonblockinggroups` : the non-blocking automata processing with Prefetching
Always-Active States (ngAP+optimization 1)
- `nonblockingpcgroups` : the non-blocking automata processing with Prefetching
Always-Active States and Prefix Memoization (ngAP+optimization 1&2)
- `nonblockingallgroups` : the non-blocking automata processing with Prefetching
Always-Active States and Prefix Memoization and Work Privatization (ngAP+optimization 1&2&3)

For more command options in `ngap` and other executables, please refer to the `-h` option.


We also provide a small NFA and an input stream to verify that the binaries are successfully built. For example, to check `ngap` in the small dataset:

```bash
ngap -a ${NGAP_ROOT}/small_dataset/apple.anml                                         \
  -i ${NGAP_ROOT}/small_dataset/inputstream.txt                                       \
  --app-name=apple --algorithm=nonblockingallgroups --input-start-pos=0               \
  --input-len=81 --split-entire-inputstream-to-chunk-size=81  --group-num=1           \
  --duplicate-input-stream=1 --unique=false --unique-frequency=10 --use-soa=false     \
  --result-capacity=54619400 --use-uvm=false --data-buffer-fetch-size=25600000        \
  --add-aan-start=256 --add-aas-interval=32 --active-threshold=10                     \
  --precompute-cutoff=-1 --precompute-depth=3 --compress-prec-table=true              \
  --report-off=false --validation=true
```

 If the build is successful, you will see the following output:

```
...
############ Validate result ############
Validation PASS!
Result(4):
0x400000005, 0x40000002f, 0x400000040, 0x40000004b,
Reference result(4):
0x400000005, 0x40000002f, 0x400000040, 0x40000004b,

ngap elapsed time: 3.6864e-05 seconds, throughput = 2.19727 MB/s
FINISHED!
```

This command will run ngAP on the provided small dataset. Additionally, a serial version of automata processing on the CPU will be executed to validate the results of the GPU version. As shown in the results, the 'apple.anml' automata reports ending positions for the 'apple' pattern in the input stream at positions: 5, 47, 64, and 75, with a state index of 4, and it passes the validation.


## 3. Run Artifact Evaluation
We provide the parameter configurations on RTX 3090 in the [config](code/scripts/configs/) folder.
You can edit application parameters in the JSON file `app_sepc_*` and schemes parameters in the JSON file `exec_config_*` under the config folder.

To run the experiments in the paper, please follow the instructions below:

```bash
${NGAP_ROOT}/scripts/run-throughput.sh    # 16 hrs
${NGAP_ROOT}/scripts/run-breakdown.sh     # 8 hrs
${NGAP_ROOT}/scripts/run-latency.sh       # 3 hrs
```

All resulting data will be stored in the `result/raw` folder, and log files will be located in `raw_results` named according to the execution date.

To generate the figures and tables as presented in the paper based on the results in the `result/raw` folder, run the following commands and you'll find the figures and tables in the `result` folder.

```bash
${NGAP_ROOT}/scripts/gen-throughput-fig13tab4.sh
${NGAP_ROOT}/scripts/gen-breakdown-fig14.sh
${NGAP_ROOT}/scripts/gen-latency-fig20tab6.sh
```

For your reference, we have included results collected on the NVIDIA RTX 3090, aw well as the figures and tables in the `ref_result` folder.

## 4. Trouble Shooting

- Building fails:

  If you encounter the following error during the compilation process:
  `fatal error: 'tbb/blocked_range.h' file not found`,
  please ensure that you have TBB installed.

- Running fails:

  If you encounter the following error during the execution process:
  `CUDA error: an illegal memory access was encountered`,
  it mainly because the option of input stream is not set correctly. Please ensure that the input stream is set to the correct path and the input stream length and number is set to the correct value.
  
- Validation fails:

    Please remove the `-quick-validation` option from the command line and try again. If the validation continues to fail, it might be due to a buffer overflow caused by too many states. To address this, consider rebuilding the project with a larger buffer size using the following CMake command: `cmake -DCMAKE_BUILD_TYPE=Debug -DDATA_BUFFER_SIZE=1000000000 -DRESULTS_SIZE=80000000 ..`
    You can adjust these values based on the available device memory on your GPU.
    In debug mode, the program will include assertions to check buffer overflow.


## Paper
Please refer to this paper for more details.

```
@inproceedings{asplos24ngap,
  title={ngAP: Non-blocking Large-scale Automata Processing on GPUs},
  author={Tianao Ge, Tong Zhang, and Hongyuan Liu},
  booktitle={Proceedings of the 29th International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS ’24)},
  year={2024}
}
```

## Related Code

- [HyperScan](https://github.com/intel/hyperscan)
- [GPU-NFA](https://github.com/bigwater/gpunfa-artifact.git)
- [VASim](https://github.com/jackwadden/VASim)
- [MNCaRT](https://github.com/kevinaangstadt/MNCaRT)
- [AutomataZoo](https://github.com/tjt7a/AutomataZoo)
- [ANMLZoo](https://github.com/jackwadden/ANMLZoo)
- [Regex](https://regex.wustl.edu/index48754875.html?title=Main_Page)


