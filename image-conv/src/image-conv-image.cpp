//==============================================================
// Vector Add is the equivalent of a Hello, World! sample for data parallel
// programs. Building and running the sample verifies that your development
// environment is setup correctly and demonstrates the use of the core features
// of DPC++. This sample runs on both CPU and GPU (or FPGA). When run, it
// computes on both the CPU and offload device, then compares results. If the
// code executes on both CPU and offload device, the device name and a success
// message are displayed. And, your development environment is setup correctly!
//
// For comprehensive instructions regarding DPC++ Programming, go to
// https://software.intel.com/en-us/oneapi-programming-guide and search based on
// relevant terms noted in the comments.
//
// DPC++ material used in the code sample:
// •	A one dimensional array of data.
// •	A device queue, buffer, accessor, and kernel.
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// DPC++ Example
// Image Convoluton with DPC++
//
// yluo
//
// (c) 2020-
//
#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include "dpc_common.hpp"
#if FPGA || FPGA_EMULATOR
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif

using namespace sycl;

// useful header files for image convolution
#include "utils.h"
#include "bmp-utils.h"
#include "gold.h"

using Duration = std::chrono::duration<double>;
class Timer {
 public:
  Timer() : start(std::chrono::steady_clock::now()) {}

  Duration elapsed() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(now - start);
  }

 private:
  std::chrono::steady_clock::time_point start;
};

static const char* inputImagePath = "./Images/cat.bmp";

static float gaussianBlurFilterFactor = 273.0f;
static float gaussianBlurFilter[25] = {
   1.0f,  4.0f,  7.0f,  4.0f, 1.0f,
   4.0f, 16.0f, 26.0f, 16.0f, 4.0f,
   7.0f, 26.0f, 41.0f, 26.0f, 7.0f,
   4.0f, 16.0f, 26.0f, 16.0f, 4.0f,
   1.0f,  4.0f,  7.0f,  4.0f, 1.0f};
static const int gaussianBlurFilterWidth = 5;

static float sharpenFilterFactor = 8.0f;
static float sharpenFilter[25] = {
    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f,  2.0f,  2.0f,  2.0f, -1.0f,
    -1.0f,  2.0f,  8.0f,  2.0f, -1.0f,
    -1.0f,  2.0f,  2.0f,  2.0f, -1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
static const int sharpenFilterWidth = 5;

static float edgeSharpenFilterFactor = 1.0f;
static float edgeSharpenFilter[9] = {
    1.0f,  1.0f, 1.0f,
    1.0f, -7.0f, 1.0f,
    1.0f,  1.0f, 1.0f};
static const int edgeSharpenFilterWidth = 3;

static float vertEdgeDetectFilterFactor = 1.0f;
static float vertEdgeDetectFilter[25] = {
     0,  0, -1.0f,  0,  0,
     0,  0, -1.0f,  0,  0,
     0,  0,  4.0f,  0,  0,
     0,  0, -1.0f,  0,  0,
     0,  0, -1.0f,  0,  0};
static const int vertEdgeDetectFilterWidth = 5;

static float embossFilterFactor = 1.0f;
static float embossFilter[9] = {
    2.0f,  0.0f,  0.0f,
    0.0f, -1.0f,  0.0f,
    0.0f,  0.0f, -1.0f};
static const int embossFilterWidth = 3;

enum filterList
{
    GAUSSIAN_BLUR,
    SHARPEN,
    EDGE_SHARPEN,
    VERT_EDGE_DETECT,
    EMBOSS,
    FILTER_LIST_SIZE
};
//static const int filterSelection = VERT_EDGE_DETECT;
//static const int filterSelection = GAUSSIAN_BLUR;
//static const int filterSelection = EDGE_SHARPEN;
static const int filterSelection = EMBOSS;

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)
#define DEVICE_NAME_LEN 128
static char dev_name[DEVICE_NAME_LEN];

#if 1
class IntMatrix {
  public:
    size_t row, column;
    std::vector<int> elements;
  
  IntMatrix(size_t r, size_t c, int initVal) {
    row = r;
    column = c;
    elements = std::vector<int>(r*c, initVal);
  }
  int e(size_t r, size_t c) {
    return elements[r*column+c];
  }
};
// matrice shapes for this example.
constexpr size_t a_rows = 200;
constexpr size_t a_columns = 400;
constexpr size_t b_columns = 600;

// matrix A size: a_rows x a_columns
// matrix B size: a_columns x b_columns
// matrices C an D size: a_rows x b_columns

#endif

float4 *pixel2rgba(float *image_in, size_t ImageRows, size_t ImageCols, image_channel_order chan_order)
{
    // allocate spaces
    float4 *ret = (float4 *)malloc(ImageRows*ImageCols*sizeof(float4));
    if (chan_order == image_channel_order::luminance) {
      return ret;
    }
    else {
      std::cout << "ERROR: unknown image channel order" << std::endl;
      free(ret);
      return NULL;
    }
}
//************************************
// Image Convolution in DPC++ on device: 
//************************************
void ImageConv(queue &q, void *image_in, void *image_out, float *filter_in, 
    const size_t FilterWidth, size_t ImageRows, size_t ImageCols) 
{
    // We create images for the input and output data.
    // Images objects are created from a host pointer together with provided 
    // channel order and channel type. 
    // image_in is a host side buffer of size ImageRows x ImageCols
    // each data item in image_in is float, representing a pixel 
    // In the example file cat.bmp, each pixel is of 8-bit color, so we just 
    // use "r" as channel order which replicates the value in all R component 
    // in the image object 
    // The channel type is set as fp32
    //
    image<2> srcImage(image_in, image_channel_order::r, image_channel_type::fp32,
                        range<2>(ImageCols, ImageRows));
    
    image<2> dstImage(image_out, image_channel_order::r, image_channel_type::fp32,
                        range<2>(ImageCols, ImageRows));

    //for(int i=0; i<ImageRows; i++) {
    //  for(int j=0; j<ImageCols; j++)
    //    std::cout << "image_out[" << i << "," << j << "]=" << (float *)image_out[i*ImageCols+j] << std::endl;
    //}

    // Create the range object for the pixel data managed by the image.
    range<2> num_items{ImageCols, ImageRows};

    // Create buffers that hold the filter shared between the host and the devices.
    buffer<float, 1> filter_buf(filter_in, range<1>(FilterWidth*FilterWidth));

    // Submit a command group to the queue by a lambda function that contains the
    // data access permission and device computation (kernel).
    q.submit([&](handler &h) {
      // Create an accessor to image with access permission: read, write or
      // read/write. The accessor is a way to access the memory in the image.
      // When accessing images, the accessor element type is used to specify 
      // how the image should be read from or written to. 
      // It can be either int4, uint4 or float4. 
      accessor<float4, 2, access::mode::read, access::target::image> srcPtr(
        srcImage, h);

      // Another way to get access is to call get_access() member function 
      //auto dstPtr = dstImage.get_access<float4, access::mode::write>(h);
      accessor<float4, 2, access::mode::write, access::target::image> dstPtr(
        dstImage, h);

      // Samplers are used to specify the way in which the coordinates map to
      // a particular pixel in the image. 
      // In our example, we specify 
      //  (1) the sampler will not use normalized co-ordinates, 
      //  (2) addresses outside the image bounds should clamp to the edge of the image 
      //  (3) and floating-point co-ordinates should take the nearest pixel's data,
      //      rather that applying (for example) a linear filter.
      sampler mysampler(coordinate_normalization_mode::unnormalized,
                    addressing_mode::clamp, filtering_mode::nearest);

      // create an accessor to the filter
      auto f_acc = filter_buf.get_access<access::mode::read>(h);

      // Use parallel_for to run image convolution in parallel on device. This
      // executes the kernel.
      //    1st parameter is the number of work items.
      //    2nd parameter is the kernel, a lambda that specifies what to do per
      //    work item. The parameter of the lambda is the work item id.
      // DPC++ supports unnamed lambda kernel by default.
      h.parallel_for(num_items, [=](id<2> item) 
      { 

        // get row and col of the pixel assigned to this work item
        int column = item[0];
        int row = item[1];

        // Half the width of the filter is needed for indexing memory later 
        int halfWidth = (int)(FilterWidth/2);

        // Iterator for the filter */
        int filterIdx = 0;

        // Each work-item iterates around its local area based on the
        // size of the filter 
        // Coordinates for accessing the image
        int2 coords;

        // store the new pixel
        float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};

        /* Iterate the filter rows */
        for(int i = -halfWidth; i <= halfWidth; i++)
        {
          coords[1] = row + i;
          /* Iterate over the filter columns */
          for(int j = -halfWidth; j <= halfWidth; j++)
          {
            coords[0] = column + j;

            // images are read using coordinates and a sampler
            // which can interpolate image data between pixels. 
            // 
            // The returned float4 represents R,G,B,Alpha
            // for our blackwhite BMP file, these values are the same because we
            // set "luminance" as the image channel order
            //
            // we use only one element in the vector
            float4 pixel = srcPtr.read(coords, mysampler);
            sum[0] += pixel[0] * f_acc[filterIdx++];
          }
        }

        // Copy the data to the output image 
        coords[0] = column;
        coords[1] = row;
        // FIXME, just write test data
        //sum = {4444.0f, 3333.0f, 2222.0f, 1111.0f};
        // Images are written to in a similar fashion without a sampler.
        dstPtr.write(coords, sum);
      }
    );
  });
}


int main() {
  // Create device selector for the device of your interest.
#if FPGA_EMULATOR
  // DPC++ extension: FPGA emulator selector on systems without FPGA card.
  INTEL::fpga_emulator_selector d_selector;
#elif FPGA
  // DPC++ extension: FPGA selector on systems with FPGA card.
  INTEL::fpga_selector d_selector;
#else
  // The default device selector will select the most performant device.
  //default_selector d_selector;
  cpu_selector d_selector;
#endif

  float *hInputImage;
  float *hOutputImage;

  int imageRows;
  int imageCols;
  int i;

  /* Set the filter here */
  cl_int filterWidth;
  float filterFactor;
  float *filter;

  // Query about the platform
  /*
  unsigned number = 0;
  auto myPlatforms = platform::get_platforms();
  // loop through the platforms to poke into
  for (auto &onePlatform : myPlatforms) {
    std::cout << ++number << " found .." << std::endl << "Platform: " 
    << onePlatform.get_info<info::platform::name>() <<std::endl;
    // loop through the devices
    auto myDevices = onePlatform.get_devices();
    for (auto &oneDevice : myDevices) {
      std::cout << "Device: " 
      << oneDevice.get_info<info::device::name>() <<std::endl;
    }
  }
*/

  // set conv filter
  switch (filterSelection)
  {
    case GAUSSIAN_BLUR:
      filterWidth = gaussianBlurFilterWidth;
      filterFactor = gaussianBlurFilterFactor;
      filter = gaussianBlurFilter;
      break;
    case SHARPEN:
      filterWidth = sharpenFilterWidth;
      filterFactor = sharpenFilterFactor;
      filter = sharpenFilter;
      break;
    case EDGE_SHARPEN:
      filterWidth = edgeSharpenFilterWidth;
      filterFactor = edgeSharpenFilterFactor;
      filter = edgeSharpenFilter;
      break;
    case VERT_EDGE_DETECT:
      filterWidth = vertEdgeDetectFilterWidth;
      filterFactor = vertEdgeDetectFilterFactor;
      filter = vertEdgeDetectFilter;
      break;
    case EMBOSS:
      filterWidth = embossFilterWidth;
      filterFactor = embossFilterFactor;
      filter = embossFilter;
      break;
    default:
      printf("Invalid filter selection.\n");
      return 1;
  }

  for (int i = 0; i < filterWidth*filterWidth; i++)
  {
    filter[i] = filter[i]/filterFactor;
  }

  /* Read in the BMP image */
  hInputImage = readBmpFloat(inputImagePath, &imageRows, &imageCols);
  printf("imageRows=%d, imageCols=%d\n", imageRows, imageCols);
  printf("filterWidth=%d, \n", filterWidth);
  /* Allocate space for the output image */
  hOutputImage = (float *)malloc( imageRows*imageCols * sizeof(float) );
  for(i=0; i<imageRows*imageCols; i++)
    hOutputImage[i] = 1234.0;


  Timer t;

  try {
    queue q(d_selector, dpc_common::exception_handler);

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";

    // Image convolution in DPC++
    ImageConv(q, hInputImage, hOutputImage, filter, filterWidth, imageRows, imageCols);
  } catch (exception const &e) {
    std::cout << "An exception is caught for image convolution.\n";
    std::terminate();
  }

  std::cout << t.elapsed().count() << " seconds\n";

  /* Save the output bmp */
  printf("Output image saved as: cat-filtered.bmp\n");
  writeBmpFloat(hOutputImage, "cat-filtered.bmp", imageRows, imageCols,
          inputImagePath);

  /* Verify result */
  float *refOutput = convolutionGoldFloat(hInputImage, imageRows, imageCols,
    filter, filterWidth);

  writeBmpFloat(refOutput, "cat-filtered-ref.bmp", imageRows, imageCols,
          inputImagePath);

  bool passed = true;
  for (i = 0; i < imageRows*imageCols; i++) {
    if (fabsf(refOutput[i]-hOutputImage[i]) > 0.001f) {
        //printf("%f %f\n", refOutput[i], hOutputImage[i]);
        passed = false;
    }
  }
  if (passed) {
    printf("Passed!\n");
    std::cout << "Image Convolution successfully completed on device.\n";
  }
  else {
    printf("Failed!\n");
  }

  return 0;
}
