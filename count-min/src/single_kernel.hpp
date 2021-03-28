//
// This file contains all of the FPGA device code for the single-kernel design
//

#ifndef __SINGLE_KERNEL_HPP__
#define __SINGLE_KERNEL_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "count_min_sketch.hpp"

using namespace sycl;

// Forward declare the kernel names to reduce name mangling
class K;

// submit the kernel for the single-kernel design
template<typename T>
event SubmitSingleWorker(queue &q, T *in_ptr, T *out_ptr, size_t count,
  buffer<int,2> C_buf, buffer<int,2> hashes_buf) {
  auto e = q.submit([&](handler& h) {
    // create accessors to buffers (can be in two ways)
    accessor C_buf_acc(C_buf, h, read_write);
    auto hashes_buf_acc = hashes_buf.get_access<access::mode::read>(h);

    h.single_task<K>([=]() [[intel::kernel_args_restrict]] {
      // using a host_ptr class tells the compiler that this pointer lives in
      // the hosts address space
      host_ptr<T> in(in_ptr);
      host_ptr<T> out(out_ptr);

      // allocate local memory for counter arrays and hash tables
      int local_mem_C[NUM_D][NUM_W];
      int local_mem_hashes[NUM_D][2];

      // load the latest counter array and hash tables from global memory
      for(auto i=0; i<NUM_D ; i++) {
        for(auto j=0; j<NUM_W; j++)
          local_mem_C[i][j] = C_buf_acc[i][j];
        local_mem_hashes[i][0] = hashes_buf_acc[i][0];
        local_mem_hashes[i][1] = hashes_buf_acc[i][1];
      }

      for (size_t i = 0; i < count; i++) {
        // get the data
        T data = *(in + i);
        // update hash tables in CM sketch
        cms_update(local_mem_C, local_mem_hashes, data, 1);

        //*(out + i) = data;
      }

      // write updated counter array and hash tables to global memory
      for(auto i=0; i<NUM_D ; i++) {
        for(auto j=0; j<NUM_W; j++)
          C_buf_acc[i][j] = local_mem_C[i][j];
      }

    });
  });

  return e;
}

#endif /* __SINGLE_KERNEL_HPP__ */
