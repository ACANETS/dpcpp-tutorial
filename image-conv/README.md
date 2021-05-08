# `sf-filter` Sample

The Sobel_Feldman operator runs two image convolution processes on an input image to detect the gradients in both the horizontal and vertical directions. This application is a very simple algorithm used to detect vertical and horizontal lines in imagery.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04
| Hardware                          | Skylake with GEN9 or newer, Intel(R) Programmable Acceleration Card with Intel(R) Arria(R) 10 GX FPGA
| Software                          | Intel&reg; oneAPI DPC++ Compiler (beta)

## Purpose

The Sobel-Feldman operator was designed as an introduction to programming in DPC++. Two different implementations of the Sobel-Feldman operator are designed in this application. The first uses the Image-Conv sample in this repo. The other implementation has two kernels running in parallel to compute the gradients for each pixel, and the forwards the pixel values, via pipes, to a consumer task. The consumer task calculates the Root Mean Sum for each pixel, and then sends the output image back to the host.

## Key Implementation Details

This 'sf-filter' example uses DPC++ buffers to store image data and a kernel function to perform convolution operation on the image data, which then gets piped to a consumer kernel function. The output of this kernel function is sent back to the host, where it gets written to a bmp file. A few utility functions, such as reading image data from bmp files or writing new image data to bmp files, are provided.

## Future Work

The current implementation of this application only builds for the FPGA Emulator target. The current use of the PipeArray uses about 10x more ALUTs than any FPGA on Devcloud can support, however limiting this number causes the gradient producer kernels to deadlock the program. Some work should be done in the future to implement a better PipeArray.

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
    ./sf-filter.fpga_emu
    ```
or
    ```
    ./sf-filter.fpga
    ```
or
    ```
    ./sf-filter.fpga_profile
    ```

### Application Parameters
There are no command line parameters for this sample. The input images are provided in "./Images" directory.

### Example of Output
<pre>
$ ./sf-filter.fpga_emu
1 found >>
Platform: Intel(R) FPGA Emulation Platform for OpenCL(TM)
Device: Intel(R) FPGA Emulation Device
2 found >>
Platform: Intel(R) FPGA SDK for OpenCL(TM)
Device: pac_s10 : Intel PAC Platform (pac_ec00000)
3 found >>
Platform: Intel(R) OpenCL
Device: Intel(R) Xeon(R) Platinum 8256 CPU @ 3.80GHz
4 found >>Platform: SYCL host platform
Device: SYCL host device
Reading input image from ./Images/cat.bmp
offset = 1078
width = 1080
height = 720
bits per pixel = 8
imageRows=720, imageCols=1080
Running on device: Intel(R) FPGA Emulation Device                                                                                                                                                                                            Enqueuing producer 0...
Enqueuing producer 1...
Enqueuing consumer...
Horizontal kernel compute time:  93.7039 ms.
Vertical kernel compute time:  55.4169 ms.
Consumer kernel compute time:  9.0035 ms.
Total compute time:  197.006 ms.
Total Image_Conv compute time:  393.862 ms.
Output image saved as ./Images/filtered_cat.bmp.
</pre>
