# dpcpp-tutorial
Lab exercises on Data Parallel Computing and DPC++ for Intel FPGA. 

## Module 1: Introduction to Data Parallel Computing and DPC++

We use [matrix multiplication](https://github.com/ACANETS/dpcpp-tutorial/tree/master/matrix-multi) as a starting point to introduce data parallel computing. (A Surprise ;)
This example shows different programming APIs for defining kernel functions that compute on partitions of data. We also demonstrate the usage of analysis and optimization tools on the designes target for Intel FPGAs.

## Module 2: Image Convolution

[Image convolution](https://github.com/ACANETS/dpcpp-tutorial/tree/master/image-conv) is an image processing operator to add pixel values to their neighbors weighted by a kernel (aka convolutional matrix). It can generate interesting effects on the original image such as edge detection or blurring, and has inherent parallelism. We will show how to implement the convolution operation in DPC++ and optimize the design.

## Module 3: Word Count 

[Word count](https://github.com/ACANETS/dpcpp-tutorial/tree/master/word-count) is an example application leveraging Map-Reduce, which decomposes the data with a "split-apply-combine" strategy for data analysis. We show how to design a kernel function to count the number of words in a document using Map-Reduce.

## Module 4: Count-Min Sketch

Count-Min Sketch utilizes a probabilistic data structure as a frequency table of events in a stream of data such as packets in network flows. Characteristics of the streaming data such as ``heavy-hitters'' can be calculated using such a streaming algorithm. We use Count-Min sketch to illustrate how to design data parallel programs to implement the algorithm.

# Acknowledgement
The development of the curriculum modules in this tutorial is sponsored by Intel Corporation under a MindShare grant in 2020-2021.
