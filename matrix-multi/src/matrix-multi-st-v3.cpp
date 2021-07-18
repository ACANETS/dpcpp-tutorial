//==============================================================
// DPC++ Example
//
// Matrix Multiplication with DPC++
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
#include <cmath>
#include "dpc_common.hpp"
#if FPGA || FPGA_EMULATOR || FPGA_PROFILE
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif

using namespace sycl;

// matrice shapes for this example.
// A: a_rows x a_columns
// B: a_columns x b_columns
// C,Sum: a_rows x b_columns
//constexpr size_t a_rows = 896;
constexpr size_t a_rows = 800;
constexpr size_t a_columns = 1600;
constexpr size_t b_columns = 3200;

//#define BLOCK_SIZE 64
#define BLOCK_SIZE 16
// define FPGA onchip memory banks and widths
#define NUM_BANKS 16
//#define BANK_WIDTH 512
#define BANK_WIDTH 64
#if (BLOCK_SIZE*BLOCK_SIZE) != (NUM_BANKS*BANK_WIDTH/4)
#error 'FPGA onchip memory needs correct number of banks and depth'
#endif

class MMstv3;
class MMstv3_cplusd;

void MatrixMulti_st_v3(queue &q, float (*matrix_a)[a_columns], float (*matrix_b)[b_columns], 
  float (*matrix_c)[b_columns], float (*matrix_d_parallel)[b_columns]) {
#if FPGA || FPGA_PROFILE
  double kernel_time_ns, total_kernel_time_ns = 0;
#endif

  std::cout << "MatrixMultiplication using single_task() v3." << std::endl;

  // Create the range object for the arrays managed by the buffer.
  range<2> num_items{a_rows, b_columns};

  // Create buffers that hold the data shared between the host and the devices.
  // The buffer destructor is responsible to copy the data back to host when it
  // goes out of scope.
  buffer<float, 2> a_buf(reinterpret_cast<float *>(matrix_a), range(a_rows, a_columns));
  buffer<float, 2> b_buf(reinterpret_cast<float *>(matrix_b), range(a_columns, b_columns));
  buffer<float, 2> c_buf(reinterpret_cast<float *>(matrix_c), num_items);
  buffer<float, 2> sum_buf(reinterpret_cast<float *>(matrix_d_parallel), num_items);

  auto step = 0;
  // Submit a command group to the queue by a lambda function that contains the
  // data access permission and device computation (kernel).
  event e = q.submit([&](handler &h) {
    // Create an accessor for each buffer with access permission: read, write or
    // read/write. The accessor is a mean to access the memory in the buffer.
    auto a = a_buf.get_access<access::mode::read, access::target::global_buffer>(h);
    auto b = b_buf.get_access<access::mode::read, access::target::global_buffer>(h);
    auto d = sum_buf.get_access<access::mode::read_write, access::target::global_buffer>(h);
      

    // A kernel that is executed on one thread using NDRange(1,1,1) is enqueued 
    // using the cl::sycl::single_task API:
    //   single_task<typename kernel_lambda_name>([=](){});
    h.single_task<MMstv3>([=]() [[intel::kernel_args_restrict]]
    { 
      size_t row, col, m, n, k;
      float s = 0;
          // allocate local memory to hold a block of data from A, B
	        [[intel::numbanks(NUM_BANKS), intel::bankwidth(BANK_WIDTH)]] float local_mem_a[BLOCK_SIZE][BLOCK_SIZE];
	        [[intel::numbanks(NUM_BANKS), intel::bankwidth(BANK_WIDTH)]] float local_mem_b[BLOCK_SIZE][BLOCK_SIZE];
	        [[intel::numbanks(NUM_BANKS), intel::bankwidth(BANK_WIDTH)]] float local_mem_d[BLOCK_SIZE][BLOCK_SIZE];

      for(int i=0; i < a_rows*a_columns/(BLOCK_SIZE*BLOCK_SIZE); i++) {
        // the block indices of the block in A 
        auto block_row_a = i / (a_columns/BLOCK_SIZE);
        auto block_col_a = i % (a_columns/BLOCK_SIZE);
        // we need to calculate dot-product with all the blocks in B where
        // the row number is equal to block_col_a
        auto block_row_b = block_col_a;
        for(int j=0; j<b_columns/BLOCK_SIZE; j++)
        { 

          // load blocks of data to local memory from global memory
          for (m=0; m < BLOCK_SIZE; m++)
            for ( n=0; n < BLOCK_SIZE; n++)
            {
              local_mem_a[m][n] = a[block_row_a*BLOCK_SIZE + m][block_col_a*BLOCK_SIZE + n];
              local_mem_b[m][n] = b[block_row_b*BLOCK_SIZE + m][j*BLOCK_SIZE + n];
              local_mem_d[m][n] = d[block_row_a*BLOCK_SIZE + m][j*BLOCK_SIZE + n];
            }  
          // element-wise multiplication and accumulation
          for (m=0; m < BLOCK_SIZE; m++)
            for ( n=0; n < BLOCK_SIZE; n++) {
              s = 0;
	      // #pragma unroll
              for (k=0; k < BLOCK_SIZE; k++)
                s += local_mem_a[m][k] * local_mem_b[k][n]; 
              // add to Matrix D
              // the corresponding row and col in D
              row = block_row_a * BLOCK_SIZE + m;
              col = j * BLOCK_SIZE + n;
              //d[row][col] += s;
              local_mem_d[m][n] += s;
            }

          // write d back to global memory
          for (m=0; m < BLOCK_SIZE; m++)
            for ( n=0; n < BLOCK_SIZE; n++)
            {
                d[block_row_a*BLOCK_SIZE + m][j*BLOCK_SIZE + n] = local_mem_d[m][n] ;
            }  
        } // for j
      } // for i
    }); // h
  }); // event e
#if FPGA || FPGA_PROFILE
      // Query event e for kernel profiling information
      // (blocks until command groups associated with e complete)
      kernel_time_ns =
        e.get_profiling_info<info::event_profiling::command_end>() -
        e.get_profiling_info<info::event_profiling::command_start>();

      // Report profiling info
      // step++;
      //std::cout << "step " << step <<" Kernel compute time:  " << kernel_time_ns * 1e-6 << " ms\n";

      total_kernel_time_ns += kernel_time_ns;
#endif

  //
  e = q.submit([&](handler &h) {
    auto c = c_buf.get_access<access::mode::read, access::target::global_buffer>(h);
    auto d = sum_buf.get_access<access::mode::write, access::target::global_buffer>(h);
      
    h.single_task<MMstv3_cplusd>([=]() [[intel::kernel_args_restrict]]
    { 
      size_t row, col, m, n, k;

      // load blocks of data to local memory from global memory
      for (m=0; m < a_rows; m++)
        for ( n=0; n < b_columns; n++)
          d[m][n] += c[m][n];
    });
  }); // event e
#if FPGA || FPGA_PROFILE
  // Query event e for kernel profiling information
  // (blocks until command groups associated with e complete)
  kernel_time_ns =
    e.get_profiling_info<info::event_profiling::command_end>() -
    e.get_profiling_info<info::event_profiling::command_start>();

  // Report profiling info
  //std::cout << " Kernel compute time:  " << kernel_time_ns * 1e-6 << " ms\n";

  total_kernel_time_ns += kernel_time_ns;

  // Report profiling info as it takes multiple steps
  std::cout << " Total Kernel compute time:  " << total_kernel_time_ns * 1e-6 << " ms\n";
#endif
}


//************************************
// Demonstrate matrix multiplication both in sequential on CPU and in parallel on device.
//************************************
int main() {
  // Create device selector for the device of your interest.
#if FPGA_EMULATOR
  // DPC++ extension: FPGA emulator selector on systems without FPGA card.
  INTEL::fpga_emulator_selector d_selector;
#elif defined(FPGA) || defined(FPGA_PROFILE)
  // DPC++ extension: FPGA selector on systems with FPGA card.
  INTEL::fpga_selector d_selector;
#else
  // The default device selector will select the most performant device.
  default_selector d_selector;
  //cpu_selector d_selector;
#endif

  // Query about the platform
  //
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
  std::cout << std::endl;

  // Create matrices with row, column and initial value 
  // to store the input and output data.
  float(*A)[a_columns] = new float[a_rows][a_columns];
  // Intialize values
  for (int i = 0; i < a_rows; i++)
    for (int j = 0; j < a_columns; j++) A[i][j] = 1.0;

  float(*B)[b_columns] = new float[a_columns][b_columns];
  // Intialize values
  for (int i = 0; i < a_columns; i++)
    for (int j = 0; j < b_columns; j++) B[i][j] = 2.0;

  float(*C)[b_columns] = new float[a_rows][b_columns];
  // Intialize values
  for (int i = 0; i < a_rows; i++)
    for (int j = 0; j < b_columns; j++) C[i][j] = 3.0;

  float(*sum_sequential)[b_columns] = new float[a_rows][b_columns];
  float(*sum_stv3)[b_columns] = new float[a_rows][b_columns];

  // Intialize values
  for (int i = 0; i < a_rows; i++)
    for (int j = 0; j < b_columns; j++) {
      sum_sequential[i][j] = 0.0;
      sum_stv3[i][j] = 0.0;
    }

  std::cout << "Matrix A size: " << a_rows << "," << a_columns << std::endl;
  std::cout << "Matrix B size: " << a_columns << "," << b_columns << std::endl;
  std::cout << "Matrices C, D size: " << a_rows << "," 
              << b_columns << std::endl;

#ifndef FPGA_PROFILE
  // Start the timer (using std::chrono)
  dpc_common::TimeInterval exec_time;    

  // Compute the sum of two arrays in sequential for validation.
  std::cout << "computing on host..." << std::endl;
  for (size_t i = 0; i < a_rows; i++)
    for (size_t j = 0; j < b_columns; j++) {
      sum_sequential[i][j] = C[i][j];
      for (size_t k = 0; k < a_columns; k++)
        sum_sequential[i][j] += A[i][k] * B[k][j];
    }

  double host_time_s = exec_time.Elapsed();
  std::cout << "host compute time " << host_time_s * 1000 << " ms\n";
#endif

  try {
    queue q(d_selector, dpc_common::exception_handler, 
            property::queue::enable_profiling{});

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";

    // Matrix multiplication in DPC++
    MatrixMulti_st_v3(q, A, B, C, sum_stv3);

#ifndef FPGA_PROFILE
    // Verify that the two arrays are equal.
    for (size_t i = 0; i < a_rows; i++)
      for (size_t j = 0; j < b_columns; j++) 
        if( abs(sum_sequential[i][j] - sum_stv3[i][j]) > 0.001) {
          std::cout << "not equal" << std::endl;
          std::cout << i << " " << j << " " << sum_sequential[i][j] 
          << " " << sum_stv3[i][j] << std::endl;
          return -1;
        }
    std::cout << "Matrix multiplication successfully completed on device.\n";
#endif

  } catch (exception const &e) {
    std::cout << "An exception is caught for matrix multiplication.\n";
    std::terminate();
  }

  return 0;
}
