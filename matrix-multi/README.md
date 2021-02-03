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
The basic DPC++ implementation explained in the code includes device selector, parallel_for(), single_task(), loop unrolling.

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
/*:
1. Clean the `matrix-muti` program using:
    ```
    make clean -f Makefile.fpga
    ```

2. Based on your requirements, you can perform the following:
   * Build and run for FPGA emulation using the following commands:
    ```
    make fpga_emu -f Makefile.fpga
    make run_emu -f Makefile.fpga
    ```
    * Build and run for FPGA hardware.
      **NOTE:** The hardware compilation takes a long time (1~2 hours) to complete.
    ```
    make hw -f Makefile.fpga
    make run_hw -f Makefile.fpga
    ```
    * Generate static optimization reports for design analysis. Path to the reports is `matrix_multi_report.prj/reports/report.html`
    ```
    make report -f Makefile.fpga
    ```
*/

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
