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
#include "dpc_common.hpp"
#if FPGA || FPGA_EMULATOR || FPGA_PROFILE
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif

using namespace sycl;

// matrice shapes for this example.
// A: a_rows x a_columns
// B: a_columns x b_columns
// C,Sum: a_rows x b_columns
constexpr size_t a_rows = 800;
constexpr size_t a_columns = 1600;
constexpr size_t b_columns = 3200;

class MMstv1;

void MatrixMulti_st_v1(queue &q, float (*matrix_a)[a_columns], float (*matrix_b)[b_columns], 
  float (*matrix_c)[b_columns], float (*matrix_d_parallel)[b_columns]) {

  std::cout << "MatrixMultiplication using single_task() v1." << std::endl;

  // Create the range object for the arrays managed by the buffer.
  range<2> num_items{a_rows, b_columns};

  // Create buffers that hold the data shared between the host and the devices.
  // The buffer destructor is responsible to copy the data back to host when it
  // goes out of scope.
  buffer<float, 2> a_buf(reinterpret_cast<float *>(matrix_a), range(a_rows, a_columns));
  buffer<float, 2> b_buf(reinterpret_cast<float *>(matrix_b), range(a_columns, b_columns));
  buffer<float, 2> c_buf(reinterpret_cast<float *>(matrix_c), num_items);
  buffer<float, 2> sum_buf(reinterpret_cast<float *>(matrix_d_parallel), num_items);

  // Submit a command group to the queue by a lambda function that contains the
  // data access permission and device computation (kernel).
  event e = q.submit([&](handler &h) {
    // Create an accessor for each buffer with access permission: read, write or
    // read/write. The accessor is a mean to access the memory in the buffer.
    auto a = a_buf.get_access<access::mode::read, access::target::global_buffer>(h);
    auto b = b_buf.get_access<access::mode::read, access::target::global_buffer>(h);
    auto c = c_buf.get_access<access::mode::read, access::target::global_buffer>(h);

    // The sum_accessor is used to store (with write permission) the sum data.
    auto sum = sum_buf.get_access<access::mode::write>(h);
  
    // Use single_task to run vector addition in parallel on device. This
    // executes the kernel.
    // A kernel that is executed on one thread using NDRange(1,1,1) is enqueued 
    // using the cl::sycl::single_task API:
    //   single_task<typename kernel_lambda_name>([=](){});
    h.single_task<MMstv1>([=]() [[intel::kernel_args_restrict]]
      { 
        size_t row, col;
        for (row = 0; row < a_rows; row++)
          for (col = 0; col < b_columns; col++) {
            float s = 0;
            #pragma unroll 2
            for (size_t k = 0; k < a_columns; k++)
              s += a[row][k] * b[k][col]; 
            sum[row][col]  = c[row][col] + s;
          }
      });
  });

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
  float(*sum_stv1)[b_columns] = new float[a_rows][b_columns];

  // Intialize values
  for (int i = 0; i < a_rows; i++)
    for (int j = 0; j < b_columns; j++) {
      sum_sequential[i][j] = 0.0;
      sum_stv1[i][j] = 0.0;
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
    MatrixMulti_st_v1(q, A, B, C, sum_stv1);

#ifndef FPGA_PROFILE
    // Verify that the two arrays are equal.
    for (size_t i = 0; i < a_rows; i++)
      for (size_t j = 0; j < b_columns; j++) 
        if( abs(sum_sequential[i][j] - sum_stv1[i][j]) > 0.0001) {
          std::cout << "not equal" << std::endl;
          std::cout << i << " " << j << " " << sum_sequential[i][j] 
          << " " << sum_stv1[i][j] << std::endl;          return -1;
        }
    std::cout << "Matrix multiplication successfully completed on device.\n";
#endif

  } catch (exception const &e) {
    std::cout << "An exception is caught for matrix multiplication.\n";
    std::terminate();
  }

  return 0;
}
