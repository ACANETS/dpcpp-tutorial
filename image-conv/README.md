# `image-conv` Sample

Image Convolution computes on the pixels and their neighbor pixels in an image using a weighted kernel (aka convolutional matrix). It is very useful in creating different effects on the original image such as edge detection and blurring. 

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04
| Hardware                          | Skylake with GEN9 or newer, Intel(R) Programmable Acceleration Card with Intel(R) Arria(R) 10 GX FPGA
| Software                          | Intel&reg; oneAPI DPC++ Compiler (beta)  

## Purpose

We use image-conv to show how to port existing application to data parallel programs in DPC++. We show the programming with image objects in SYCL although DPC++ currently only supports image on a CPU device, not on a GPU or FPGA device. Therefore, we include an implementation of image-conv using DPC++ buffers and use this version for experiments with FPGAs. We also include a few examples for analyzing and optimizing the design. In addition we show how to use compilation reports to analyze a design and how to perform run-time profiling. A few design options are compared side by side in terms of programming APIs, resource utilization, area and performance.

## Key Implementation Details 

This 'image-conv' example uses DPC++ buffers to store image data and a kernel function to perform convolution operation on the image data, and then writes the new image to a bmp file. The new image data are compared with those from a convolution function implemented on the host CPU as a golden standard.  A few utility functions, such as reading image data from bmp files or writing new image data to bmp files, are provided. 

We include two implementations of image-conv kernels: buffer objects based and image objects based. Buffer objects are general memory objects that can be used to store arbitrary data structure, whereas image objects are opaque data types specifically for image data and related image processing operations.

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

*A Makefile is still maintained in the directory, however, the usage of it is disencouraged.*

## Running the Sample

The executables (for emulation or for FPGA hardware) can be found in the build directory. Use the file name(s) to executable the samples. For example
    ```
    ./image-conv.fpga_emu
    ```
or
    ```
    ./image-conv.fpga
    ```
or
    ```
    ./image-conv.fpga_profile
    ```

### Application Parameters
There are no command line parameters for this sample. The input images are provided in "./Images" directory.

### Example of Output
<pre>
$ ./image-conv.fpga_emu 
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

Reading input image from ./Images/cat.bmp
offset = 1078
width = 1080
height = 720
bits per pixel = 8
imageRows=720, imageCols=1080
filterWidth=3, 
Running on device: Intel(R) FPGA Emulation Device
0.212566 seconds
Output image saved as: cat-filtered.bmp
Passed!
Image Convolution successfully completed on device.
</pre>

## Recorded Lectures

A series of recorded lectures are provided to introduce the important concepts in this image-conv example for FPGAs. The videos can be found at the [DPC++ Tutorial playlist](https://youtube.com/playlist?list=PLZ9YeF_1_vF8RqYPNpHToklJcDRoVocU4) on Youtube and are linked individually below. 

[Introduction to Image Convolution](https://youtu.be/O_-sXNy23mw)

[Image Objects in DPC++](https://youtu.be/Kb46lMLsve0)

