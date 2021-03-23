//
// This file contains all of the FPGA device code for the single-kernel design
//

#ifndef __SINGLE_KERNEL_HPP__
#define __SINGLE_KERNEL_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "count_min_sketch.hpp"

using namespace sycl;

#define NUM_BANKS 8
#define BANK_WIDTH 16

// Forward declare the kernel names to reduce name mangling
class K;

// submit the kernel for the single-kernel design
template<typename T>
event SubmitSingleWorker(queue &q, T *in_ptr, T *out_ptr, size_t count) {
  auto e = q.submit([&](handler& h) {
    h.single_task<K>([=]() [[intel::kernel_args_restrict]] {
      // using a host_ptr class tells the compiler that this pointer lives in
      // the hosts address space
      host_ptr<T> in(in_ptr);
      host_ptr<T> out(out_ptr);

	  [[intel::numbanks(NUM_BANKS), intel::bankwidth(BANK_WIDTH)]] int counter_array[NUM_D][NUM_W];

      for (size_t i = 0; i < count; i++) {
        // do a simple copy - more complex computation can go here
        T data = *(in + i);
        // update hash tables in CM sketch
        //cms_update();

        *(out + i) = data;
      }
    });
  });

  return e;
}

#endif /* __SINGLE_KERNEL_HPP__ */
