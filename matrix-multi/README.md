# `matrix-multiplication` Sample

We use matrix multiplication example to show how to design data parallel programs in DPC++. Using it, we demonstrate how "thinking-in-parallel" helps take advantage of the parallelism in a task and implement it in DPC++. We also include a few examples for analyzing and optimizing the design. In addition we show how to use compilation reports to analyze a design and how to perform run-time profiling. A few design options are compared side by side in terms of programming APIs, resource utilization, area and performance.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04
| Hardware                          | Skylake with GEN9 or newer, Intel(R) Programmable Acceleration Card with Intel(R) Arria(R) 10 GX FPGA
| Software                          | Intel&reg; oneAPI DPC++ Compiler (beta)  
  
The compilation and run-time profiling are performed on Intel FPGA DevCloud, a free development platform.

## Purpose

We use matrix-multi example to demonstrate how DPC++ supports kernel functions that define how computation is carried out on a partition of the whole dataset or task. We also show how to use tools to analyze and optimize the design. 

## Key Implementation Details 
The DPC++ implementations explained in the several versions covers basic concepts of DPC++ programming such as device selector, parallel_for(), single_task(), loop unrolling. In the st-v3 version, we introduce the tile concept that relies on the usage of local memory to reduce the cost of accessing global memory.

## License  
This code sample is licensed under MIT license. 


## Building the `matrix-multi` Program for Intel(R) FPGA

### On a Linux* System

**The project uses CMake. Perform the following steps to build different targets.** 

```
    mkdir build
    cd build
    cmake ..
    make
    make report
    make fpga
    make profile
```
* make : by default, the emulation executables are built.
* make report : generate static report on the FPGA resource utilization of the designs.
* make fpga : generate FPGA binary files for the designs. Will take a couple of hours.
* make profile : generate FPGA binary to be used in run-time profiling. Will take a couple of hours.

*A Makefile is still maintained in the directory, however, the usage of it is disencouraged.*

## Running the Sample

The executables (for emulation or for FPGA hardware) can be found in the build directory. Use the file name(s) to executable the samples. For example
    ```
    ./matrix-multi-para.fpga_emu
    ```

### Application Parameters
There are no editable parameters for this sample.

### Example of Output

* emulation on a Linux platform.
<pre>
1 found ..
Platform: Intel(R) FPGA Emulation Platform for OpenCL(TM)
Device: Intel(R) FPGA Emulation Device
2 found ..
Platform: Intel(R) OpenCL
Device: Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz
3 found ..
Platform: Intel(R) OpenCL HD Graphics
Device: Intel(R) Graphics [0x5912]
4 found ..
Platform: Intel(R) Level-Zero
Device: Intel(R) Graphics [0x5912]
5 found ..
Platform: SYCL host platform
Device: SYCL host device

computing on host...
1.83809 seconds
Running on device: Intel(R) FPGA Emulation Device
Matrix A size: 800,1000
Matrix B size: 1000,2000
Matrices C, D size: 800,2000
MatrixMultiplication using parallel_for().
0.183899 seconds
Matrix multiplication successfully completed on device.

</pre>

## Recorded Lectures

A series of recorded lectures are provided to introduce the important concepts about DPC++ programming for FPGAs. The videos can be found at the [DPC++ Tutorial playlist](https://youtube.com/playlist?list=PLZ9YeF_1_vF8RqYPNpHToklJcDRoVocU4) on Youtube and are linked individually below. 

[Introduction to DPC++](https://youtu.be/F2DWVuJRvfM)

[How to Think "in Parallel"?](https://youtu.be/3DTYEBSrj-U)

[FPGA Design Concepts](https://youtu.be/dLGY7_ql1H8)

[Design Analysis (I): FPGA Early Image](https://youtu.be/zpPbn0eOCg8)

[Design Analysis (II): Runtime Profiling](https://youtu.be/q2KZvAqhN_s)
