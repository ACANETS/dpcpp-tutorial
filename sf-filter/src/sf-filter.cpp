//***************************************************************************************
// DPC++ Example
//
// Sobel-Feldman Filter with DPC++
//
// Author: Matthew Cloutier
//
//***************************************************************************************
#include <CL/sycl.hpp>
#include <array>
#include <cmath>
#include <iostream>
#include "dpc_common.hpp"
#if FPGA || FPGA_EMULATOR || FPGA_PROFILE
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif

using namespace sycl;

// useful headers for image processing
#include "utils.h"
#include "bmp-utils.h"

// Debug flag for print statements
static const bool debug = true;

static const int sobelFilterWidth = 3;
static float verticalSobelFilter[9] = {
   3.0f, 0.0f,  -3.0f,
  10.0f, 0.0f, -10.0f,
   3.0f, 0.0f,  -3.0f
};

static float horizontalSobelFilter[9] = {
   3.0f,  10.0f,  3.0f,
   0.0f,   0.0f,  0.0f,
  -3.0f, -10.0f, -3.0f
};

//
// Image Info
//
static const char* inputImagePath = "../image-conv/Images/cat.bmp";
static const char* output_filename = "../image-conv/Images/filtered_cat.bmp";

#define IMAGE_SIZE (720*1080)
constexpr size_t array_size = IMAGE_SIZE;
typedef std::array<float, array_size> FloatArray;

// **************************************************************************************
// Image Convolution in DPC++ on device:
//  Copied from image-conv.cpp and modified for specific Sobel-Feldman filtering
// **************************************************************************************
void ImageConv(queue &q, float *image_in, float *image_out,
                  float *filter_in, const size_t FilterWidth,
                  const size_t ImageRows, const size_t ImageCols)
{
  //
  // Create buffers for the input and output data
  //
  buffer<float, 1> image_in_buf(image_in, range<1>(ImageRows * ImageCols));
  buffer<float, 1> image_out_buf(image_out, range<1>(ImageRows * ImageCols));

  //
  // Create the range object for the pixel data
  //
  range<2> num_items {ImageRows, ImageCols};

  //
  // Create buffers that hold the filter shared between the host and the devices.
  //
  buffer<float, 1> filter_buf(filter_in, range<1>(FilterWidth*FilterWidth));

  //
  // Compute the filter width (intentionally truncate)
  //
  int halfFilterWidth = (int)FilterWidth/2;

  //
  // Submit a command group to the queue by a lambda function that contains the data access
  // permission and device computation (kernel)
  //
  q.submit([&](handler &h)
  {
    //
    // Create an accessor to the buffers with access permission.
    //
    auto sourcePtr = image_in_buf.get_access<access::mode::read>(h);
    auto destPtr = image_out_buf.get_access<access::mode::write>(h);
    auto filterPtr = filter_buf.get_access<access::mode::read>(h);

    //
    // Use parallel_for to run the image convolution in parallel on device.
    // This executes the kernel.
    //    1st parameter is the number of work items.
    //    2nd parameter is the kernel, the lambda to specify what to do per work item.
    //      The parameter of the lambda is the work item id.
    h.parallel_for(num_items, [=](id<2> item)
    {
      //
      // Get the cow and col of the pixel assigned to this work item
      //
      int row = item[0];
      int col = item[1];

      //
      // Half the width of the filter is needed for indexing memory later
      //
      int halfWidth = (int)(FilterWidth / 2);

      //
      // Iterator for the filter
      //
      int filterIdx = 0;

      // Each work item iterates around its local are based on the size of the filter.
      float sum = 0.0f;

      //
      // Apply the filter to the pixel neigborhood
      //
      for (int k = -halfFilterWidth; k <= halfFilterWidth; k++)
      {
        for (int l = -halfFilterWidth; l <= halfFilterWidth; l++)
        {
          // Indices used to access the image
          int r = row + k;
          int c = col + l;

          // Handle out of bounds  locations by clamping to the border pixel
          r = (r < 0) ? 0 : r;
          c = (c < 0) ? 0 : c;
          r = (r >= ImageRows) ? ImageRows - 1 : r;
          c = (c >= ImageCols) ? ImageCols - 1 : c;

          sum += sourcePtr[r * ImageCols + c] *
            filterPtr[(k + halfFilterWidth) * FilterWidth + (l + halfFilterWidth)];

        }
      }

      //
      // Save off the new pixel value
      //
      destPtr[row * ImageCols + col] = sum;

    });
  });

}

int main()
{
  // Create device selector for the device of your interest.
  #if FPGA_EMULATOR
    // DPC++ extension: FPGA emulator selector on systems without FPGA card.
    INTEL::fpga_emulator_selector d_selector;
  #elif FPGA || FPGA_PROFILE
    // DPC++ extension: FPGA selector on systems with FPGA card.
    INTEL::fpga_selector d_selector;
  #else
    // The default device selector will select the most performant device.
    default_selector d_selector;
  #endif

  float *hInputImage;
  float *hOutputImage;
  float *vOutputImage;
  float *outputImage;

  int imageRows;
  int imageCols;

  float *filter;

  #ifndef FPGA_PROFILE
    // Query some platform information
    unsigned number = 0;
    auto myPlatforms = platform::get_platforms();

    //loop through the platforms to poke
    for (auto &onePlatform: myPlatforms)
    {
      std::cout << ++number << " found >>" << std::endl;
      std::cout << "Platform: " << onePlatform.get_info<info::platform::name>() << std::endl;

      // Loop through devices
      auto myDevices = onePlatform.get_devices();
      for (auto &oneDevice : myDevices)
      {
        std::cout << "Device: " << oneDevice.get_info<info::device::name>() << std::endl;
      }
    }

    std::cout << std::endl;
  #endif

  //
  // Read in the image
  //
  hInputImage = readBmpFloat(inputImagePath, &imageRows, &imageCols);
  if (debug)
  {
    printf("imageRows=%d, imageCols=%d\n", imageRows, imageCols);
  }

  //
  // Allocate space for the resulting images
  //
  hOutputImage = (float*) malloc(imageRows * imageCols * sizeof(float));
  vOutputImage = (float*) malloc(imageRows * imageCols * sizeof(float));
  outputImage = (float*) malloc(imageRows * imageCols * sizeof(float));

  try
  {
    queue q(d_selector, dpc_common::exception_handler);

    //
    // Print device information used in the kernel
    //
    std::cout << "Running on device: " <<
      q.get_device().get_info<info::device::name>() << std::endl;

    //
    // Run the horizontal line filter, then the vertical line filter
    //
    filter = horizontalSobelFilter;
    ImageConv(q, hInputImage, hOutputImage, filter, sobelFilterWidth, imageRows, imageCols);

    filter = verticalSobelFilter;
    ImageConv(q, hInputImage, vOutputImage, filter, sobelFilterWidth, imageRows, imageCols);

  }
  catch(const std::exception& e)
  {
    std::cerr << "Caught the following error executing ImageConv:" << std::endl;
    std::cerr << e.what() << std::endl;
    std::terminate();
  }

  //
  // No errors runing kernel, so combine the two filtered images
  //
  for (int r = 0; r < imageRows; r++)
  {
    for (int c = 0; c < imageCols; c++)
    {
      outputImage[r * imageCols + c] = std::sqrt(
        std::pow(hOutputImage[r * imageCols + c], 2) +
         std::pow(vOutputImage[r * imageCols + c], 2));
    }
  }

  //
  // Save the output bmp
  //
  printf("Output image saved as %s.\n", output_filename);
  writeBmpFloat(outputImage, output_filename, imageRows, imageCols, inputImagePath);

  // TODO MRC - Verify results?

  return 0;

}