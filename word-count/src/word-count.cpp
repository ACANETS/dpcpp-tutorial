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
// number of keywords to search
#define NUM_KEYWORDS 4

constexpr unsigned MAX_WG_SIZE = 16;
constexpr unsigned CHAR_PER_WORKITEM = 1024;

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

size_t text_size;

//************************************
// Word Count in DPC++ on device: 
//************************************
void string_search(queue &q, uint32_t total_num_workitems, uint32_t n_wgroups, 
  int wgroup_size, std::vector<char4> pattern, char* text, int chars_per_item, uint32_t* global_result) 
{
#if FPGA || FPGA_PROFILE
  double total_kernel_time_ns = 0;
#endif

  char4 keywords[NUM_KEYWORDS];
  for(int k = 0; k < NUM_KEYWORDS ; k++){
    keywords[k] = pattern[k];
  }

  // buffers for device
  buffer<char,1> text_buf(text, range<1>(text_size));
  buffer<uint32_t, 1> global_result_buf(global_result, range<1>(NUM_KEYWORDS));

  std::cout << std::endl << "n_wgroups = " << n_wgroups << std::endl;
  std::cout << "wgroup_size = " << wgroup_size << std::endl;

  auto n_steps = (int)(total_num_workitems + n_wgroups*wgroup_size -1) / 
    (n_wgroups*wgroup_size);

  auto step = 0;

  while(step < n_steps ) {
    
    event e = q.submit([&] (handler& h) {
    // allocate local memory
    // to allow each workgroup has a local memory space of int32_t*NUM_KEYWORDS
    // for maintaining a set of keyword counters for all the workitems in a workgroup
    accessor <uint32_t, 1,
      access::mode::read_write,
      access::target::local>
    local_mem(range<1>(NUM_KEYWORDS), h);

    // point to global memory where the final results are stored
    auto global_mem = global_result_buf.get_access<access::mode::read_write>(h);

    // point to global memory where the text are stored
    auto text_mem = text_buf.get_access<access::mode::read>(h);
    auto text_max_len = text_size;
    // use nd_range to specify kernels' global size and local size.
    // in this case, we use one-dimensional nd_range
    // the first range object specifies the number of (total) work items per dimension
    // the second range object specifies the number of work items in a work group
    h.parallel_for<class reduction_kernel>(
      nd_range<1>(n_wgroups * wgroup_size, wgroup_size),
      [=] (nd_item<1> item) 
      [[intel::max_work_group_size(1, 1, MAX_WG_SIZE), 
        cl::reqd_work_group_size(1,1,MAX_WG_SIZE),
        intel::num_simd_work_items(MAX_WG_SIZE)]] 
      {

        // initialize local data
        group<1> g = item.get_group();
        size_t group_id = g.get_id();
        size_t local_id = item.get_local_id(0);

        if (local_id == 0) {
          local_mem[0] = 0;
          local_mem[1] = 0;
          local_mem[2] = 0;
          local_mem[3] = 0;
        }
        item.barrier(sycl::access::fence_space::local_space);         

        // In each step, each work item will process char_per_item characters
        // Prior to this step, there are many characters processed already 
        int item_offset = step * n_wgroups * wgroup_size * chars_per_item 
                          + group_id * wgroup_size * chars_per_item 
                          + local_id * chars_per_item;

        /* Iterate through characters in text */
        for(int i=item_offset; i<item_offset + chars_per_item; i++) {
          // check bounds of text buffer
          if(i > text_max_len-4)
            break;
          //load one four-character word
          char4 text_word;
          text_word.load(0, text_mem.get_pointer()+i);
          for(int k = 0; k < NUM_KEYWORDS ; k++){
            if (text_word.x() == keywords[k].x() &&
                text_word.y() == keywords[k].y() &&
                text_word.z() == keywords[k].z() &&
                text_word.w() == keywords[k].w()
              )
            {
              // we need to increment the count (in local mem) ATOMICALLY for keywords[k]
              local_atomic_ref<uint32_t>(local_mem[k])++;
            }
          }
        }

        item.barrier(sycl::access::fence_space::local_space); 

        if( local_id == 0) {
          global_atomic_ref<uint32_t>(global_mem[0]) += local_mem[0];
          global_atomic_ref<uint32_t>(global_mem[1]) += local_mem[1];
          global_atomic_ref<uint32_t>(global_mem[2]) += local_mem[2];
          global_atomic_ref<uint32_t>(global_mem[3]) += local_mem[3];
        }

      }); // parallel_for
    }); // q.submit
#if FPGA || FPGA_PROFILE
    // Query event e for kernel profiling information
    // (blocks until command groups associated with e complete)
    double kernel_time_ns =
      e.get_profiling_info<info::event_profiling::command_end>() -
      e.get_profiling_info<info::event_profiling::command_start>();

    // Report profiling info
    std::cout << "step " << step <<" Kernel compute time:  " << kernel_time_ns * 1e-6 << " ms\n";

    total_kernel_time_ns += kernel_time_ns;
#endif
    step++;
  } // while

  std::cout<<"total "<<step<<" steps completed.\n";

#if FPGA || FPGA_PROFILE
    // Report profiling info as it takes multiple steps
    std::cout << " Total Kernel compute time:  " << total_kernel_time_ns * 1e-6 << " ms\n";
#endif
}


int main(int argc, char **argv) {
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
  //cpu_selector d_selector;
#endif

  uint32_t result[4] = {0, 0, 0, 0};
  // we search for four key words: "that", "with", "have", "from"
  std::vector<char4> pattern;
  pattern.push_back({'t','h','a','t'});
  pattern.push_back({'w','i','t','h'});
  pattern.push_back({'h','a','v','e'});
  pattern.push_back({'f','r','o','m'});

  FILE *text_handle;
  char *text;
  int chars_per_item;
  int n_local_results;

  /* Read text file and place content into buffer */
  if (argc != 2)
    text_handle = fopen(TEXT_FILE, "r");
  else
    text_handle = fopen(argv[1], "r");
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
  std::cout << "file size = " << text_size << " bytes " << std::endl;

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
    auto num_cmpunit =
        dev.get_info<cl::sycl::info::device::max_compute_units>();
    std::cout << "num of compute units (reported)= " << num_cmpunit << std::endl;

    num_cmpunit = 1 < num_cmpunit ? 1 : num_cmpunit;
    std::cout << "num of compute units (set as)= " << num_cmpunit << std::endl;

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
        || local_mem_size < (wgroup_size * sizeof(int32_t)*NUM_KEYWORDS))
    {
        throw "Device doesn't have enough local memory!";
    }
    else
    {
      std::cout << "local_mem_size = " << local_mem_size << std::endl;
    }
    
    auto global_mem_size = dev.get_info<info::device::global_mem_size>();
    std::cout << "global_mem_size = " << global_mem_size << std::endl;

    auto total_num_workitems = (int)(text_size + CHAR_PER_WORKITEM - 1) / CHAR_PER_WORKITEM;
    auto num_groups = num_cmpunit;
    std::cout << "chars_per_item = " << CHAR_PER_WORKITEM << std::endl;
    std::cout << "total_num_workitems = " << total_num_workitems << std::endl;
    std::cout << "num_groups = " << num_groups << std::endl;

    // Word count in DPC++
    string_search(q, total_num_workitems, num_groups, wgroup_size, pattern, text, 
      CHAR_PER_WORKITEM, result);
  
  } catch (exception const &e) {
    std::cout << "An exception is caught for word count.\n";
    std::terminate();
  }

  // display final results in global memory
  for(int i=0; i < NUM_KEYWORDS; i++)
    std::cout << "keyword " << pattern[i][0]<<pattern[i][1]<<pattern[i][2]<<pattern[i][3] 
    << " appears " << result[i] << " times" << std::endl;

  return 0;
}
