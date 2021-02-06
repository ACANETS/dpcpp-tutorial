//==============================================================
// DPC++ Example
//
// Image Convoluton with DPC++
//
// Author: Yan Luo
//
// Copyright ©  2020-
//
// MIT License
//
#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include "dpc_common.hpp"
#if FPGA || FPGA_EMULATOR || FPGA_PROFILE
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
static const int filterSelection = EDGE_SHARPEN;
//static const int filterSelection = EMBOSS;

#define IMAGE_SIZE (720*1080)
constexpr size_t array_size = IMAGE_SIZE;
typedef std::array<float, array_size> FloatArray;

//************************************
// Image Convolution in DPC++ on device: 
//************************************
void ImageConv_v1(queue &q, float *image_in, float *image_out, float *filter_in, 
    const size_t FilterWidth, const size_t ImageRows, const size_t ImageCols) 
{

    // We create buffers for the input and output data.
    //
    buffer<float, 1> image_in_buf(image_in, range<1>(ImageRows*ImageCols));
    buffer<float, 1> image_out_buf(image_out, range<1>(ImageRows*ImageCols));

    //for(int i=0; i<ImageRows; i++) {
    //  for(int j=0; j<ImageCols; j++)
    //    std::cout << "image_out[" << i << "," << j << "]=" << (float *)image_out[i*ImageCols+j] << std::endl;
    //}

    // Create the range object for the pixel data.
    range<2> num_items{ImageRows, ImageCols};

    // Create buffers that hold the filter shared between the host and the devices.
    buffer<float, 1> filter_buf(filter_in, range<1>(FilterWidth*FilterWidth));

    /* Compute the filter width (intentionally truncate) */
    int halfFilterWidth = (int)FilterWidth/2;

    // Submit a command group to the queue by a lambda function that contains the
    // data access permission and device computation (kernel).
    q.submit([&](handler &h) {
      // Create an accessor to buffers with access permission: read, write or
      // read/write. The accessor is a way to access the memory in the buffer.
      accessor srcPtr(image_in_buf, h, read_only);

      // Another way to get access is to call get_access() member function 
      auto dstPtr = image_out_buf.get_access<access::mode::write>(h);

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
        int row = item[0];
        int col = item[1];

        // Half the width of the filter is needed for indexing memory later 
        int halfWidth = (int)(FilterWidth/2);

        // Iterator for the filter */
        int filterIdx = 0;

        // Each work-item iterates around its local area based on the
        // size of the filter 

        float sum = 0.0f;

        /* Apply the filter to the neighborhood */
        for (int k = -halfFilterWidth; k <= halfFilterWidth; k++) 
        {
          for (int l = -halfFilterWidth; l <= halfFilterWidth; l++)
          {
              /* Indices used to access the image */
              int r = row+k;
              int c = col+l;
              
              /* Handle out-of-bounds locations by clamping to
              * the border pixel */
              r = (r < 0) ? 0 : r;
              c = (c < 0) ? 0 : c;
              r = (r >= ImageRows) ? ImageRows-1 : r;
              c = (c >= ImageCols) ? ImageCols-1 : c;       
              
              sum += srcPtr[r*ImageCols+c] *
                    f_acc[(k+halfFilterWidth)*FilterWidth + 
                        (l+halfFilterWidth)];
          }
        }
         
        /* Write the new pixel value */
        dstPtr[row*ImageCols+col] = sum;

      }
    );
  });
}


int main() {
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

  int imageRows;
  int imageCols;
  int i;

  /* Set the filter here */
  cl_int filterWidth;
  float filterFactor;
  float *filter;

#ifndef FPGA_PROFILE
  // Query about the platform
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
  std::cout<<std::endl;
#endif

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
    ImageConv_v1(q, hInputImage, hOutputImage, filter, filterWidth, imageRows, imageCols);
  } catch (exception const &e) {
    std::cout << "An exception is caught for image convolution.\n";
    std::terminate();
  }

  std::cout << t.elapsed().count() << " seconds\n";

  /* Save the output bmp */
  printf("Output image saved as: cat-filtered.bmp\n");
  writeBmpFloat(hOutputImage, "cat-filtered.bmp", imageRows, imageCols,
          inputImagePath);

#ifndef FPGA_PROFILE
  /* Verify result */
  float *refOutput = convolutionGoldFloat(hInputImage, imageRows, imageCols,
    filter, filterWidth);

  writeBmpFloat(refOutput, "cat-filtered-ref.bmp", imageRows, imageCols,
          inputImagePath);

  bool passed = true;
  for (i = 0; i < imageRows*imageCols; i++) {
    if (fabsf(refOutput[i]-hOutputImage[i]) > 0.001f) {
        printf("%f %f\n", refOutput[i], hOutputImage[i]);
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
#endif

  return 0;
}
