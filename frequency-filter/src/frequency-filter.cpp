//***************************************************************************************
// DPC++ Example
//
// Digital Frequency Filter with DPC++
//
// Author: Matthew Cloutier
//
//***************************************************************************************

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include <cmath>
#include <complex>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bmp-utils.h"

#include "dpc_common.hpp"
#include "oneapi/mkl.hpp"

using namespace sycl;

#if !defined(MKL_COMPLEX8)
#define MKL_Complex8 std::complex<double>
#endif

#if !defined(M_PI)
#define M_PI 3.14159265358979323846
#endif

// Debug flag for print statements
static const bool debug = true;

//
// Image Info
//
static const char* inputImagePath = "./input/cat.bmp";
static const char* check_inputImagePath = "./output/check-original.bmp";

static const char* fft_filename = "./output/fft.bmp";
static const char* output_filename = "./output/output.bmp";

#define IMAGE_SIZE (720*1080)

// Kernel Functions
/*sycl::event fft_2d(matrix_r &image_out, matrix_r &image_in, descriptor_real &fft2d, sycl::queue &queue);
sycl::event interpolation(matrix_r &image_out, matrix_r &image_in, sycl::queue &main_queue, const sycl::vector_class<sycl::event> &deps);
sycl::event ifft_2d(matrix_r &image_out, descriptor_complex &ifft2d, sycl::queue &main_queue, const sycl::vector_class<sycl::event> &deps);
*/

typedef oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
                                     oneapi::mkl::dft::domain::REAL>
        descriptor_real;

// Support Functions
template <typename T>
void die(std::string err, T param);
void die(std::string err);

int main(int argc, char* argv[])
{
  bool need_help = false;
  std::string filter_type = "lowpass";

  // Dimensions of image
  int p = 400;
  int q = 400;

  // parse the command line arguments
  for (int i = 1; i < argc; i++)
  {
    std::string arg(argv[i]);

    if (arg == "--help" || arg == "-h")
    {
      need_help = true;
    }
    else
    {
      std::string str_after_equals = arg.substr(arg.find("=") + 1);

      if (arg.find("--filter=") == 0)
      {
        filter_type = str_after_equals.c_str();
        // TODO - Error checking.
        if (false)
        {
          need_help = true;
          std::cout << "ERROR: Invalid filter type '" << arg << "'\n";
        }
      }
      else
      {
        std::cout << "WARNING: ignoring unknown argument '" << arg << "'\n";
      }
    }
  }

  // print help is asked
  if (need_help)
  {
    std::cout << "USAGE: "
              << "./frequency-filter "
              << "[--filter={low-pass, high-pass, band-pass}]\n";
    return 0;
  }

  try
  {
    // device selector
    #if defined(FPGA_EMULATOR)
      INTEL::fpga_emulator_selector selector;
    #else
      INTEL::fpga_selector selector;
    #endif

    auto exception_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const &e) {
                std::cout << "Caught asynchronous SYCL exception:" << std::endl
                          << e.what() << std::endl;
            }
        }
    };

    property_list prop_list { property::queue::enable_profiling() };

    queue main_queue(selector, exception_handler, prop_list);

    int imageRows;
    int imageCols;
    int imageDepth;

    //
    // Read in the image
    std::cout << "Reading original image from " << inputImagePath << std::endl;
    double* inputImage;
    double* outputImage;
    MKL_Complex8* filteredImage;
    double* fftImage;
    double* shifted_fftImage;

    inputImage = readBmpDouble(inputImagePath, &imageRows, &imageCols, &imageDepth);

    filteredImage = (MKL_Complex8 *) malloc((imageCols/2 + 1) * imageRows * sizeof(MKL_Complex8));
    fftImage = (double *) malloc(imageRows * imageCols * sizeof(double));
    shifted_fftImage = (double *) malloc(imageRows * imageCols * sizeof(double));

    //
    // Step 1 - Compute the forward fft of the image
    descriptor_real fft2d({imageRows, imageCols});

    // Configure FFT Descriptor
    std::int64_t in_strides[3] = {0, imageCols, 1};
    std::int64_t out_strides[3] = {0, (imageCols / 2), 1};
    double scale = (double)1.0 / (imageRows * imageCols);

    fft2d.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, in_strides);
    fft2d.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, out_strides);

    fft2d.set_value(oneapi::mkl::dft::config_param::FORWARD_SCALE, scale);
    fft2d.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
    fft2d.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);

    fft2d.commit(main_queue);

    const sycl::vector_class<sycl::event> deps = {};
    auto step1 = oneapi::mkl::dft::compute_forward(fft2d, inputImage, filteredImage, deps);
    step1.wait();

    double min = 10000000000000000000.0;
    double max = 0;
    for (auto i = 0; i < imageRows; i++)
    {
      for (auto j = 0; j < imageCols; j++)
      {
        MKL_Complex16 val;
        if(j < imageCols/2+1)
        {
          val.real = filteredImage[i*(imageCols/2+1)+j].real();
          val.imag = filteredImage[i*(imageCols/2+1)+j].imag();
        }
        else // unpack CCE format result
        {
          if(i == 0)
          {
              val.real = filteredImage[imageCols-j].real();
              val.imag = filteredImage[imageCols-j].imag() * (-1);
          }
          else
          {
              val.real = filteredImage[(imageRows-i)*(imageCols/2+1)+imageCols-j].real();
              val.imag = filteredImage[(imageRows-i)*(imageCols/2+1)+imageCols-j].imag() * (-1);
          }
        }
        double amp = log(sqrt(val.real * val.real + val.imag * val.imag));
        fftImage[i*imageCols+j] = amp; // save amplitude of individual complex number into buffer
        if(amp < min)
            min = amp;
        if(amp > max)
            max = amp;
      }
    }

    for (auto i = 0; i < imageRows*imageCols; i++)
      fftImage[i] = 255.0 * (fftImage[i] - min) / (double)(max-min); // normalize buffer values to [0-255]

    int i0, j0;
    for (auto i = 0; i < imageRows; ++i) // shift zero-frequency component to center
    {
      for (auto j = 0; j < imageCols; ++j)
      {
        if (i < imageRows / 2)
            i0 = i + imageRows / 2;
        else
            i0 = i - imageRows / 2;
        if (j < imageCols / 2)
            j0 = j + imageCols / 2;
        else
            j0 = j - imageCols / 2;

        shifted_fftImage[i * imageDepth + j] = fftImage[i0 * imageCols + j0];
      }
    }

    writeBmpDouble(shifted_fftImage, fft_filename, imageRows, imageCols, inputImagePath);

    /*matrix_r outputImage(main_queue);
    outputImage.allocate(2 * q, 2 * 2 * q, 2 * 2 * q);
    if (!outputImage.data) {
      die("Cannot allocate memory for outputImage.");
    }

    descriptor_complex ifft2d({outputImage.h, (outputImage.w) / 2});
    auto step3 = ifft_2d(filteredImage, ifft2d, main_queue, {step1});
    step3.wait();

    std::cout << "Writing final image to " << output_filename << std::endl;
    bmp_write(output_filename, filteredImage, true);*/
  }
  catch(const std::exception& e)
  {
    std::cerr << "Caught the following error executing FrequencyFilter:" << std::endl;
    std::cerr << e.what() << std::endl;
    std::terminate();
  }

  return 0;
}

template <typename T>
void die(std::string err, T param)
{
    std::cout << "Fatal error: " << err << " " << param << std::endl;
    fflush(0);
    exit(1);
}

void die(std::string err)
{
    std::cout << "Fatal error: " << err << " " << std::endl;
    fflush(0);
    exit(1);
}
