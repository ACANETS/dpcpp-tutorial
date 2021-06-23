# `word-count` Sample

Word Count example searches through a text file and count the occurences of given keywords. On Linux, it is similar to "grep", which can output every occurence of a keyword and then count with "wc" tool.  

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04
| Hardware                          | Skylake with GEN9 or newer, Intel(R) Programmable Acceleration Card with Intel(R) Stratix(R) 10 GX FPGA
| Software                          | Intel&reg; oneAPI DPC++ Compiler (beta)  

## Purpose

We use word-count to show how to design MapReduce programs. MapReduce is a widely used design pattern that can handle large problems and big data by mapping the original problem into partitions, on which the computation results are then collected and merged. In a typical MapReduce program, the first stage is a "Map" procedure which use hash or other filtering schemes to assign data to distributed servers. These servers run the tasks in parallel, and send their results to a "Reduce" procedure, which merges (e.g. sum up) such intermediate results to generate the final results. 

In this example, we show the MapReduce design pattern is applied in word counting. Specifically for FPGA architectures, we show how to implement a reduce kernel in DPC++ and have it work with a customized map procedure. We also demonstrate the usage of atomic memory operations to eliminate race conditions and ensure the correctness of the results.

## Key Implementation Details 

This 'word-count' example first finds out the information regarding the device such as the number of compute units, the maximal number of work items in a work group, and the size of memory components. This is to reveal FPGA specific features and limitations. We then set the parameters, such as the number of workgroups and size of a workgroup, used later for kernels. 

Word-count uses DPC++ a global memory buffer to store text file data and a local memory buffer to store intermediate counters for each work group. The final counter results are stored in a global memory buffer. Since multiple work-items may be incrementing the counts at the same time, we use atomic operations (e.g. read, write) to ensure no race conditions and the correctness of the increments.

The Map procedure uses a fixed "workload" represented by "char_per_item" for each work-item. As a result, we partition the input text file into equal chunks of "text_size/MAX_WG_SIZE" bytes and assign each chunk to a work-item. 

In the Reduction kernel, a work-item initializes the counters for the work-group, and reads its assigned chunk from the text file data. It scans through the chunk byte by byte and compares with the keywords simultaneously. If a keyword is matched, the corresponding counter in the local memory will be incremented atomically. At the end of reduction, one of the work-items in a work-group will increment the global counters atomically.  

In this FPGA oriented design, we have one compute unit which is typical for FPGA. Therefore one work-group is needed and specified. We have the option to set the size of the work-group, i.e. the number of work-items in the work-group. 

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
    make fpga_profile
```
* make : by default, the emulation executables are built.
* make report : generate static report on the FPGA resource utilization of the design.
* make fpga : generate FPGA binary files for the designs. Will take a couple of hours.
* make fpga_profile : generate FPGA binary to be used in run-time profiling. Will take a couple of hours.

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
$ ./word-count.fpga_emu 
file size = 121099 bytes 
1 found ..
Platform: Intel(R) FPGA Emulation Platform for OpenCL(TM)
Device: Intel(R) FPGA Emulation Device
2 found ..
Platform: Intel(R) OpenCL
Device: Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz
3 found ..
Platform: Intel(R) Level-Zero
Device: Intel(R) Graphics [0x5912]
4 found ..
Platform: SYCL host platform
Device: SYCL host device

Running on device: Intel(R) FPGA Emulation Device
num of compute units (reported)= 8
num of compute units (set as)= 2
max work group size = 67108864
Work-group size exceed max size. Set it to 16
work item dimensions = 3
max work item sizes dim[0] = 67108864
max work item sizes dim[1] = 67108864
max work item sizes dim[2] = 67108864
max_mem_alloc_size = 12606132224
local_mem_size = 262144
global_mem_size = 50424528896
chars_per_item = 32
total_num_workitems = 3785
num_groups = 2

n_wgroups = 2
wgroup_size = 16
keyword that appears 330 times
keyword with appears 237 times
keyword have appears 110 times
keyword from appears 116 times

</pre>

## Recorded Lectures

A series of recorded lectures are provided to introduce the important concepts about this word-count example for FPGAs. The videos can be found at the [DPC++ Tutorial playlist](https://youtube.com/playlist?list=PLZ9YeF_1_vF8RqYPNpHToklJcDRoVocU4) on Youtube and are linked individually below. 

[Introduction to MapReduce and Reduction Library](https://youtu.be/c0JPxu0BgZE)

[Word-Count using MapReduce on FPGA](https://youtu.be/4HhQnUH0C6A)

