#include <assert.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <functional>
#include <string>
#include <thread>
#include <type_traits>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

#include "kernel_defs.hpp"

using namespace sycl;

// the type used
// NOTE: the tutorial assumes the use of a sycl::vec datatype (like long8).
// Therefore, 'Type' must be a sycl::vec datatype (e.g. int8, char64, etc).
using Type = float*;
#define IMAGE_SIZE (720*1080)

// the main function
int main(int argc, char* argv[])
{
  // parse command line arguments
  #if defined(FPGA_EMULATOR)
    size_t size = 108;
    size_t iterations = 1;
  #else
    size_t size = 108;
    size_t iterations = 5;
  #endif

  bool need_help = false;

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

      if (arg.find("--iterations=") == 0)
      {
        iterations = std::max(2, atoi(str_after_equals.c_str()) + 1);
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
              << "./rolling-mean-filter "
              << "[--iterations=<int>]\n";
    return 0;
  }

  // check the number of iterations
  if (iterations <= 0)
  {
    std::cerr << "ERROR: 'iterations' must be positive\n";
    std::terminate();
  }

  // print info
  std::cout << "Iterations:       " << iterations-1 << "\n";
  std::cout << "\n";

  try
  {
    // device selector
    #if defined(FPGA_EMULATOR)
      INTEL::fpga_emulator_selector selector;
    #else
      INTEL::fpga_selector selector;
    #endif

    // queue properties to enable profiling
    property_list prop_list { property::queue::enable_profiling() };

    // create the device queue
    queue q(selector, dpc_common::exception_handler, prop_list);

    // make sure the device supports USM host allocations
    device d = q.get_device();
    if (!d.get_info<info::device::usm_host_allocations>())
    {
      std::cerr << "ERROR: The selected device does not support USM host allocations\n";
      std::terminate();
    }

    //
    // Setup the USM input/outputs
    //  malloc_host allocates memory specifically in the host's address space
    auto in_restricted_usm = malloc_host<float*>(IMAGE_SIZE * size, q.get_context());
    auto out_restricted_usm = malloc_host<float*>(IMAGE_SIZE * size, q.get_context());

    // Make sure we were able to allocate space for the input and output
    if (in_restricted_usm == NULL)
    {
      std::cerr << "ERROR:\tFailed to allocate space for 'in_restricted_usm'." << std::endl;
      return 1;
    }
    if (out_restricted_usm == NULL)
    {
      std::cerr << "ERROR:\tFailed to allocate space for 'out_restricted_usm'." << std::endl;
      return 1;
    }

    // Buffer to hold latency data
    std::vector<double> usm_latency(iterations);

    for (size_t i = 0; i < iterations; i++)
    {
      usm_latency[i] = Run_Iteration<float*>(q, in_restricted_usm, out_restricted_usm, size);
    }

    // The FPGA emulator does not accurately represent the hardware performance
    // so we don't print performance results when running with the emulator
    #ifndef FPGA_EMULATOR
      // Compute the average latency across all iterations.
      // We use the first iteration as a 'warmup' for the FPGA,
      // so we ignore its results.
      double usm_avg_lat =
          std::accumulate(usm_latency.begin() + 1,
                          usm_latency.end(), 0.0) /
                         (iterations - 1);

      std::cout << "Average latency for the restricted USM kernel: "
                << usm_avg_lat << " ms\n";
    #endif

    // free the allocated host usm memory
    // note that these are calls to sycl::free()
    free(in_restricted_usm, q);
    free(out_restricted_usm, q);

  }
  catch (exception const& e)
  {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.get_cl_code() == CL_DEVICE_NOT_FOUND)
    {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }

  std::cout << "PASSED\n";
  return 0;

}

