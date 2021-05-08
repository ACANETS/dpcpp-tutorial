# Rolling Mean Filter
This application details an implementation of a Rolling Mean Filter, and was created by combining the Buffered Host Stream and Zero Copy Data Transfer tutorials from Intels FPGA Code sample github.

***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10
| Hardware                          | Intel&reg; FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix&reg; 10 SX)
| Software                          | Intel&reg; oneAPI DPC++ Compiler
| What you will learn               | How to optimally stream data between the host and device to maximize throughput
| Time to complete                  | 3 hours (On target hardware)

_Notice: SYCL USM host allocations (and therefore this tutorial) are only supported for the Intel&reg; FPGA PAC D5005 (with Intel Stratix 10 SX)_
_Notice: This tutorial demonstrates an implementation of host streaming that will be supplanted by better techniques in a future release. See the [Drawbacks and Future Work](#drawbacks-and-future-work)_

## Purpose
This application details an implementation of a Weighted Rolling Mean Filter. The application streams data from the host, to the device where it calculates the mean of an image, and applies the weightes, and back to the host. The techniques employed in this application are not specific to a CPU-FPGA system (like the one used in this tutorial); they apply to GPUs, multi-core CPUs, and other processing units.

### Key Implementation Details
In this application, we will create a design where a *Producer* (running on the CPU) produces data into USM host allocations, a *Kernel* (running on the FPGA) processes this data and produces output into host allocations, and a *Consumer* (running on the CPU) consumes the data. Data is shared between the host and FPGA device via host pointers (pointers to USM host allocations).

![](block_diagram.png)


## License
Code samples are licensed under the MIT license.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `rolling-mean` Tutorial

### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (fpga_compile, fpga_runtime:arria10, or fpga_runtime:stratix10) as well as whether to run in batch or interactive mode. For more information see the Intel&reg; oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).

When compiling for FPGA hardware, it is recommended to increase the job timeout to 12h.

### On a Linux* System

1. Generate the `Makefile` by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel&reg; FPGA PAC D5005 (with Intel Stratix&reg; 10 SX), run `cmake` using the command:
    ```
    cmake ..
   ```

2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   * Compile for emulation (fast compile time, targets emulated FPGA device):
      ```
      make fpga_emu
      ```
   * Generate the optimization report:
     ```
     make report
     ```
   * Compile for FPGA hardware (longer compile time, targets FPGA device):
     ```
     make fpga
     ```

## Examining the Reports
Locate `report.html` in the `rolling-mean.prj/reports/` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./rolling-mean.fpga_emu     (Linux)
     ```
2. Run the sample on the FPGA device:
     ```
     ./rolling-mean.fpga         (Linux)
     ```

### Example of Output

You should see the following output in the console:

1. When running on the FPGA emulator
    ```
    Iterations:       0

    Starting kernel processing.
    Kernel processing done.
    PASSED
    ```
    NOTE: The FPGA emulator does not accurately represent the performance (throughput or latency) of the kernels.

2. When running on the FPGA device
    ```
    Iterations:       4

    Starting kernel processing.
    Kernel processing done.
    Starting kernel processing.
    Kernel processing done.
    Starting kernel processing.
    Kernel processing done.
    Starting kernel processing.
    Kernel processing done.

    Average latency for the restricted USM kernel: 2830ms

    PASSED
    ```
