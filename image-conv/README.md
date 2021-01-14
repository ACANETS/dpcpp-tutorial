# `image-conv` Sample

Image Convolution

## Purpose

## Key Implementation Details 


## License  
This code sample is licensed under MIT license. 

### On a Linux* System

Perform the following steps:

1. Clean the `image-conv` program using:
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
      **NOTE:** The hardware compilation takes a long time to complete.
    ```
    make hw -f Makefile.fpga
    make run_hw -f Makefile.fpga
    ```
    * Generate static optimization reports for design analysis. Path to the reports is `image-conv_report.prj/reports/report.html`
    ```
    make report -f Makefile.fpga
    ```


## Running the Sample
### Application Parameters
There are no editable parameters for this sample.

### Example of Output
<pre>
TODO
</pre>
