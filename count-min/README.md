# `count-min` Sample

Count-Min sketch is one of the  example searches through a text file and count the occurences of given keywords. On Linux, it is similar to "grep", which can output every occurence of a keyword and then count with "wc" tool.  

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04
| Hardware                          | Skylake with GEN9 or newer, Intel(R) Programmable Acceleration Card with Intel(R) Stratix(R) 10 GX FPGA and *USM* support
| Software                          | Intel&reg; oneAPI DPC++ Compiler (beta)  

## Purpose

We use word-count to .

## Key Implementation Details 

This


## License  
This code sample is licensed under MIT license. 

### On a Linux* System

**The project uses CMake. Perform the following steps to build different targets.** 

```
    mkdir build
    cd build
    cmake ..
    make
    make report
    make fpga
```
* make : by default, the emulation executables are built.
* make report : generate static report on the FPGA resource utilization of the design.
* make fpga : generate FPGA binary files for the designs. Will take a couple of hours.

On Intel FPGA DevCloud, you can run a shell script job to launch the compilation of FPGA binary in batch mode on a node scheduled by the job scheduler, instead of logining into a specific FPGA node and then starting the compilation. This batch mode frees up the FPGA nodes from unnecessary occupancy in interactive mode. We included the job scripts (for compiling fpga_profile target) for both Arria 10 and Stratix 10, and you can modify the scripts to generate other targets.

*A Makefile is still maintained in the directory, however, the usage of it is disencouraged.*

## Running the Sample

The executables (for emulation or for FPGA hardware) can be found in the build directory. Use the file name(s) to executable the samples. For example
    ```
    ./word-count.fpga_emu
    ```
or
    ```
    ./word-count.fpga
    ```
or
    ```
    ./word-count.fpga_profile
    ```

### Application Parameters
There can be zero or up to one command line parameter for this sample. If no arguments are provided, the executable will use the default input file 'kafka.txt' which is provided in the same directory. Alternatively one can use another text file by supplying the file name as the only argument.

### Example of Output
<pre>
u31830@s001-n142:~/projects/dpcpp-tutorial/count-min/build_s10_usm$ ./count-min.fpga
on Host: CM round up w = 65536; d = 16
counter array sizes = 4194304 bytes
hash table sizes = 128 bytes
# Chunks:             64
Chunk count:          512
Total count:          32768
Iterations:           0

ERROR: The selected device does not support USM host allocations
terminate called without an active exception
Aborted
</pre>
This means that the FPGA images on this DevCloud node does not support USM. We can load a USM image with the following command.
<pre>
aocl program acl0 /opt/intel/oneapi/intelfpgadpcpp/2021.2.0/board/intel_s10sx_pac/bringup/aocxs/pac_s10_usm.aocx 
aocl program: Running program from /opt/intel/inteloneapi/intelfpgadpcpp/latest/board/intel_s10sx_pac/linux64/libexec
Program succeed. 
</pre>
