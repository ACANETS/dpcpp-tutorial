# `matrix-multiplication` Sample

We use matrix multiplication example to show how to design data parallel programs. Using it, we demonstrate how "thinking-in-parallel" helps leverage the parallelism in a task and implement it in DPC++. We also include a few examples for analyzing and optimizing the design. Specifically we show how to use compilation reports to analyze a design and how to perform run-time profiling. A few design options are compared side by side in terms of programming APIs, resource utilization, area and performance.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04, Windows 10 
| Hardware                          | Skylake with GEN9 or newer, Intel(R) Programmable Acceleration Card with Intel(R) Arria(R) 10 GX FPGA
| Software                          | Intel&reg; oneAPI DPC++ Compiler (beta)  
  
## Purpose

TODO

## Key Implementation Details 
The basic DPC++ implementation explained in the code includes device selector, parallel_for(), single_task(), loop unrolling.

## License  
This code sample is licensed under MIT license. 


## Building the `matrix-multi` Program for Intel(R) FPGA

### On a Linux* System

Perform the following steps:

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
3. The project can also be built with cmake
    ```
    mkdir build
    cd build
    cmake ..
    make
    make report
    make fpga
    ```

## Running the Sample
### Application Parameters
There are no editable parameters for this sample.

### Example of Output
<pre>
Todo
</pre>
