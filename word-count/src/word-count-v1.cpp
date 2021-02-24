//==============================================================
// DPC++ Example
//
// Word Count with DPC++
//
// Author: Yan Luo
//
// Reference
//
// SYCL memory types and programming examples:
//    https://developer.codeplay.com/products/computecpp/ce/guides/sycl-guide/memory
//
// Copyright ©  2020-2021
//
// MIT License
//
// Acknowledgement
//    This work is supported by Intel MindShare Grant 2020-2021.
//
//===============================================================
//
// Specify a Work-Group Size
// Specify a maximum or the required work-group size whenever possible. 
// The Intel® oneAPI DPC++/C++ Compiler relies on this specification to 
// optimize hardware use of the DPC++ kernel without involving excess logic.
//
// More details: 
//https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide/top/optimize-your-design/resource-use/specify-a-work-group-size.html


#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include "dpc_common.hpp"
#if FPGA || FPGA_EMULATOR || FPGA_PROFILE
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif

using namespace sycl;

#define TEXT_FILE "kafka.txt"
#define MAX_TEXT_LEN 20000000
// number of keywords to search
#define NUM_KEYWORDS 4

constexpr unsigned MAX_WG_SIZE = 16;

// templates for atomic ref operations
template <typename T>
using local_atomic_ref = ONEAPI::atomic_ref<
  T,
  ONEAPI::memory_order::relaxed,
  ONEAPI::memory_scope::work_group,
  access::address_space::local_space>;

template <typename T>
using global_atomic_ref = ONEAPI::atomic_ref<
  T,
  ONEAPI::memory_order::relaxed,
  ONEAPI::memory_scope::system,
  access::address_space::global_space>;

//************************************
// Word Count in DPC++ on device: 
//************************************
void string_search(queue &q, int n_wgroups, int wgroup_size, char16 pattern, 
  char* text, int chars_per_item, int* global_result) 
{

  char4 keywords[NUM_KEYWORDS];
  for(int k = 0; k < NUM_KEYWORDS ; k++){
    keywords[k] = {pattern[k*4], pattern[k*4+1], pattern[k*4+2], pattern[k*4+3]};
    //keywords[k] = pattern[k*4].xyzw();
  }

  // buffers for device
  buffer<char,1> text_buf(text, range<1>(MAX_TEXT_LEN));
  buffer<int, 1> global_result_buf(global_result, range<1>(NUM_KEYWORDS));

  std::cout << "here = " << pattern[0] << std::endl;
  std::cout << "n_wgroups = " << n_wgroups << std::endl;
  std::cout << "wgroup_size = " << wgroup_size << std::endl;

  event e = q.submit([&] (handler& h) {

    // prepare data accessors

    // allocate local memory
    // to allow each workgroup has a local memory space of int32_t*NUM_KEYWORDS
    // and we have total of n_wgroups work groups. 
    accessor <int, 1,
      access::mode::read_write,
      access::target::local>
    local_mem(range<1>(n_wgroups*NUM_KEYWORDS), h);

    // point to global memory where the final results are stored
    auto global_mem = global_result_buf.get_access<access::mode::read_write>(h);

    // point to global memory where the text are stored
    auto text_mem = text_buf.get_access<access::mode::read>(h);

    // use nd_range to specify kernels' global size and local size.
    // in this case, we use one-dimensional nd_range
    // the first range object specifies the number of (total) work items per dimension
    // the second range object specifies the number of work items in a work group
    h.parallel_for<class reduction_kernel>(
      nd_range<1>(n_wgroups * wgroup_size, wgroup_size),
      sycl::ONEAPI::reduction(global_mem, 0, std::plus<int>()),
      [=] (nd_item<1> item, auto &global_mems) 
      //[[INTEL::max_work_group_size(1, 1, MAX_WG_SIZE)]] 
      {

        // initialize local data
        group<1> g = item.get_group();
        size_t group_id = g.get_id();
        size_t local_id = item.get_local_id(0);
        // get_global_linear_id() can map to a linear ID even in multi-dimensional case
        // here, it has the same effect as get_global_id(0)
        size_t global_id = item.get_global_linear_id();

        if (local_id == 0) {
          local_mem[group_id*NUM_KEYWORDS] = 0;
          local_mem[group_id*NUM_KEYWORDS+1] = 0;
          local_mem[group_id*NUM_KEYWORDS+2] = 0;
          local_mem[group_id*NUM_KEYWORDS+3] = 0;
        }
        item.barrier(sycl::access::fence_space::local_space);         

        int item_offset = global_id * chars_per_item;

        /* Iterate through characters in text */
        for(int i=item_offset; i<item_offset + chars_per_item; i++) {
          for(int k = 0; k < NUM_KEYWORDS ; k++){
            //load one four-character word
            char4 text_word;
            text_word.load(k, text_mem.get_pointer()+i);
            //vec<bool, 4> cmp_result;
            //cmp_result = text_word == keywords[k];
            if (text_word.x() == keywords[k].x() &&
                text_word.y() == keywords[k].y() &&
                text_word.z() == keywords[k].z() &&
                text_word.w() == keywords[k].w()
              )
            {
              // we need to increment the count (in local mem) ATOMICALLY for keywords[k]
              local_atomic_ref<int>(local_mem[group_id*NUM_KEYWORDS+k])++;
              // FIXME local_mem[group_id*NUM_KEYWORDS+k] ++;
            }
          }
        }

        item.barrier(sycl::access::fence_space::local_space); 

#if 1
        if( local_id == 0) {
          global_atomic_ref<int>(global_mem[0]) += local_mem[group_id*NUM_KEYWORDS];
          global_atomic_ref<int>(global_mem[1]) += local_mem[group_id*NUM_KEYWORDS+1];
          global_atomic_ref<int>(global_mem[2]) += local_mem[group_id*NUM_KEYWORDS+2];
          global_atomic_ref<int>(global_mem[3]) += local_mem[group_id*NUM_KEYWORDS+3];
        }
#endif

    }); // parallel_for
  }); // q.submit

#if FPGA || FPGA_PROFILE
  // Query event e for kernel profiling information
  // (blocks until command groups associated with e complete)
  double kernel_time_ns =
    e.get_profiling_info<info::event_profiling::command_end>() -
    e.get_profiling_info<info::event_profiling::command_start>();

  // Report profiling info
  std::cout << "Kernel compute time:  " << kernel_time_ns * 1e-6 << " ms\n";
#endif

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
  //default_selector d_selector;
  cpu_selector d_selector;
#endif

  int result[4] = {0, 0, 0, 0};
  // we search for four key words: "that", "with", "have", "from"
  char16 pattern = {'t','h','a','t','w','i','t','h','h','a','v','e','f','r','o','m'};
  FILE *text_handle;
  char *text;
  size_t text_size;
  int chars_per_item;
  int n_local_results;

  /* Read text file and place content into buffer */
  text_handle = fopen(TEXT_FILE, "r");
  if(text_handle == NULL) {
      perror("Couldn't find the text file");
      exit(1);
  }
  fseek(text_handle, 0, SEEK_END);
  text_size = ftell(text_handle)-1;
  rewind(text_handle);
  text = (char*)calloc(text_size, sizeof(char));
  fread(text, sizeof(char), text_size, text_handle);
  fclose(text_handle);

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

  try {
    queue q(d_selector, dpc_common::exception_handler, 
          property::queue::enable_profiling{});

    device dev = q.get_device();

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << dev.get_info<info::device::name>() << "\n";
    auto num_groups =
        dev.get_info<cl::sycl::info::device::max_compute_units>();
    std::cout << "num of compute units = " << num_groups << std::endl;

    num_groups = 2 < num_groups ? 2 : num_groups;
    std::cout << "FORCE num of compute units = " << num_groups << std::endl;

    auto wgroup_size = dev.get_info<info::device::max_work_group_size>();
    std::cout << "max work group size = " << wgroup_size << std::endl;
    if (wgroup_size > MAX_WG_SIZE) {
      std::cout<<"Work-group size exceed max size. Set it to " << MAX_WG_SIZE << std::endl;
      wgroup_size = MAX_WG_SIZE;
    }

    auto witem_dims = dev.get_info<info::device::max_work_item_dimensions>();
    std::cout << "work item dimensions = " << witem_dims << std::endl;

    auto witem_sizes = dev.get_info<info::device::max_work_item_sizes>();
    for (auto k = 0; k < witem_dims ; k++) {
      std::cout << "max work item sizes dim[" << k << "] = " << witem_sizes[k] << std::endl;
    }

    auto max_mem_alloc_size = dev.get_info<info::device::max_mem_alloc_size>();
    std::cout << "max_mem_alloc_size = " << max_mem_alloc_size << std::endl;

    auto has_local_mem = dev.is_host()
          || (dev.get_info<info::device::local_mem_type>()
          != info::local_mem_type::none);
    auto local_mem_size = dev.get_info<info::device::local_mem_size>();
    if (!has_local_mem
        || local_mem_size < (num_groups * sizeof(int32_t)*NUM_KEYWORDS))
    {
        throw "Device doesn't have enough local memory!";
    }
    else
    {
      std::cout << "local_mem_size = " << local_mem_size << std::endl;
    }
    
    auto global_mem_size = dev.get_info<info::device::global_mem_size>();
    std::cout << "global_mem_size = " << global_mem_size << std::endl;

    // global size = number of compute units * number of work items per work group
    auto global_size = num_groups * wgroup_size;    
    chars_per_item = text_size / global_size + 1;
    std::cout << "chars_per_item = " << chars_per_item << std::endl;

    // Word count in DPC++
    string_search(q, num_groups, wgroup_size, pattern, text, 
      chars_per_item, result);
  
  } catch (exception const &e) {
    std::cout << "An exception is caught for word count.\n";
    std::terminate();
  }

  // reduce the final results in global memory
  for(int i=0; i < NUM_KEYWORDS; i++)
    std::cout << "keyword " << i << " appears " << result[i] << " times" << std::endl;


  return 0;
}
