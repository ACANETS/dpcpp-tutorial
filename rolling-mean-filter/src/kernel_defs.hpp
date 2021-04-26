///////////////////////////////////////////////////////////////////////////////
//
//  The structure of this file is modeled after restricted_usm_kernel.hpp
//  in the DPC++-FPGA Tutorial for Zero Copy Data Transfer
//
//
//
// The structure of the kernels in this design is shown in the diagram below.
// The Producer kernel reads the data from CPU memory (via PCIe), producing it
// for the RestrictedUSM via a pipe. The Worker does the computation on the
// input data and writes it to the ConsumePipe. The consumer reads the data
// from this pipe and writes the output back to the CPU memory (via PCIe).
//
//  |---------------|  |-------------|  |----------------------------------|
//  |      CPU      |  |             |  |            FPGA                  |
//  |               |  |             |  |                                  |
//  | |-----------| |  |             |  | |----------|   |---------------| |
//  | |  Producer |--->|             |--->| Producer |==>|               | |
//  | |-----------| |  |             |  | |----------|   |               | |
//  |               |  | Host Memory |  |                | RestrictedUSM | |
//  | |-----------| |  |             |  | |----------|   |               | |
//  | | Consumer  |<---|             |<---| Consumer |<==|               | |
//  | |-----------| |  |             |  | |----------|   |---------------| |
//  |               |  |             |  |                                  |
//  |---------------|  |-------------|  |----------------------------------|
//
//
// As shown in the image above and the code below, we have split this design
// into three kernels:
//    1) Producer
//    2) RestrictedUSM
//    3) Consumer
// We do this to decouple the reads/writes from/to the Host Memory over PCIe.
// Decoupling the memory accesses and using SYCL pipes with a substantial
// depth ('kPipeDepth' below) allows the kernel to be more resilient against
// stalls while reading/writing from/to the Host Memory over PCIe.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef __KERNEL_DEFS_HPP__
#define __KERNEL_DEFS_HPP__
#pragma once

#include <vector>

#include "Utils/bmp-utils.h"

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

using namespace sycl;
using namespace std::chrono;

// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class RestrictedUSM;
class Producer;
class Consumer;

// pipes
constexpr size_t kPipeDepth = 64;
template <typename T>
using ProducePipe = pipe<class ProducePipeClass, T, kPipeDepth>;
template <typename T>
using ConsumePipe = pipe<class ConsumePipeClass, T, kPipeDepth>;

static const char* inputImagePath = "./input/cat_0.bmp";

static int g_imageRows = 0;
static int g_imageCols = 0;

static float meanFilter[9] = {
    0.11f, 0.11f, 0.11f,
    0.11f, 0.11f, 0.11f,
    0.11f, 0.11f, 0.11f};

/* Compute the filter width (intentionally truncate) */
static const int FilterWidth = 3;
static const int halfFilterWidth = (int)FilterWidth/2;

const bool debug = false;

//
// The producer thread function. All production of data happens in this function
// which is run in a separate CPU thread from the launching of kernels and
// consumption of data.
//
template <typename T>
void ProducerThread(T* in_buffer, size_t size)
{
  // In our case, the Producer's job is simple. It has an input stream of data
  // ('in_stream') whose size is buffer_count * reps elements
  // (i.e. ALL of the data).
  // When signalled to, it produces buffer_count elements to 'out_ptr', which is
  // a shared variable between the Producer thread and the thread launching the
  // kernels.
  int imageRows;
  int imageCols;

  size_t rep = 0;

  while (rep < size)
  {
    std::string input_filename = "./input/cat_" + std::to_string(rep) + ".bmp";

    if (debug)
      std::cout << "Reading image " << input_filename.c_str() << std::endl;

    T in_image = readBmpFloat(input_filename.c_str(), &imageRows, &imageCols);

    if (imageRows == g_imageRows && imageCols == g_imageCols)
    {
      in_buffer[rep] = in_image;
    }
    else
    {
      std::cerr << "Image row/col size does not match the first image." << std::endl;
      std::terminate();
    }
    rep++;
  }
}


//
// The consumer thread function. All consumption of data happens in this function
// which is run in a separate CPU thread from the launching of kernels and
// production of data.
//
template <typename T>
void ConsumerThread(T* out_buffer, size_t size)
{
  // In our case, the Producer's job is simple. It has an input stream of data
  // ('in_stream') whose size is buffer_count * reps elements
  // (i.e. ALL of the data).
  // When signalled to, it produces buffer_count elements to 'out_ptr', which is
  // a shared variable between the Producer thread and the thread launching the
  // kernels.
  size_t rep = 0;

  while (rep < size)
  {
    std::string output_filename = "./output/cat_" + std::to_string(rep) + ".bmp";
    if (debug) {
      std::cout << "Output image saved as " << output_filename.c_str() << std::endl;
    }
    writeBmpFloat(out_buffer[rep], output_filename.c_str(),
                  g_imageRows, g_imageCols, inputImagePath);
    rep++;
  }
}

//
// reads the input data from the hosts memory
// and writes it to the ProducePipe
//
template <typename T>
event SubmitProducer(queue& q, T* in_data, size_t size) {
  auto e = q.submit([&](handler& h) {
    h.single_task<Producer>([=]() [[intel::kernel_args_restrict]] {
      // using a host_ptr tells the compiler that this pointer lives in the
      // hosts address space
      host_ptr<T> h_in_data(in_data);

      for (size_t i = 0; i < size; i++) {
        T data_from_host_memory = *(h_in_data + i);
        ProducePipe<T>::write(data_from_host_memory);
      }
    });
  });

  return e;
}

//
// The worker kernel in the device:
//  1) read input data from the ProducePipe
//  2) perform computation
//  3) write the output data to the ConsumePipe
//
template <typename T>
event SubmitWorker(queue& q, size_t size, int imageRows, int imageCols)
{
  auto e = q.submit([&](handler& h)
  {
    h.single_task<RestrictedUSM>([=]() [[intel::kernel_args_restrict]]
    {
      for (size_t i = 0; i < size; i++)
      {
        T data = ProducePipe<T>::read();

        T value = data;

        for (int row = 0; row < imageRows; row++)
        {
          for (int col = 0; col < imageCols; col++)
          {
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
                  r = (r >= imageRows) ? imageRows-1 : r;
                  c = (c >= imageCols) ? imageCols-1 : c;

                  sum += data[r*imageCols+c] *
                        meanFilter[(k+halfFilterWidth)*FilterWidth + (l+halfFilterWidth)];
              }
            }

            /* Write the new pixel value */
            value[row*imageCols+col] = sum;
          }
        }

        ConsumePipe<T>::write(value);
      }
    });
  });

  return e;
}

//
// reads output data from the device via ConsumePipe
// and writes it to the hosts memory
//
template <typename T>
event SubmitConsumer(queue& q, T* out_data, size_t size)
{
  auto e = q.submit([&](handler& h)
  {
    // using a host_ptr tells the compiler that this pointer lives in the
    // hosts address space
    host_ptr<T> h_out_data(out_data);

    h.single_task<Consumer>([=]() [[intel::kernel_args_restrict]]
    {
      for (size_t i = 0; i < size; i++)
      {
        T data_to_host_memory = ConsumePipe<T>::read();
        *(h_out_data + i) = data_to_host_memory;
      }
    });
  });

  return e;
}

template <typename T>
double Run_Iteration(queue& q, T* in, T* out, size_t size)
{
  // Read the first image sepatately.
  // This allows us to set the global image rows/cols so that
  // we can add error checking for reading subsequent images.
  T test_image = readBmpFloat(inputImagePath, &g_imageRows, &g_imageCols);

  // start the timer
  auto start = high_resolution_clock::now();

  // start the Producer in a new thread (starts in the ProducerThread<T> function)
  std::thread producer_thread(ProducerThread<T>, in, size);

  // wait for producer thread to finish
  producer_thread.join();

  std::cout << "Starting kernel processing." << std::endl;

  // start the kernels
  auto worker_event = SubmitWorker<T>(q, size, g_imageRows, g_imageCols);
  auto producer_event = SubmitProducer<T>(q, in, size);
  auto consumer_event = SubmitConsumer<T>(q, out, size);

  // wait for all the kernels to finish
  q.wait();
  std::cout << "Kernel processing done." << std::endl;

  // start the Consumer in a new thread (starts in the ConsumerThread<T> function)
  std::thread consumer_thread(ConsumerThread<T>, out, size);

  // wait for consumer thread to finish
  consumer_thread.join();

  // stop the timer
  auto end = high_resolution_clock::now();
  duration<double, std::milli> diff = end - start;

  return diff.count();
}

#endif /* __KERNEL_DEFS_HPP__ */