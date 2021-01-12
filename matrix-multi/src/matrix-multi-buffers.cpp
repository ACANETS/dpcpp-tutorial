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
#if FPGA || FPGA_EMULATOR
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif

using namespace sycl;

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

// matrice shapes for this example.
// A: a_rows x a_columns
// B: a_columns x b_columns
// C,Sum: a_rows x b_columns
constexpr size_t a_rows = 200;
constexpr size_t a_columns = 400;
constexpr size_t b_columns = 600;

void MatrixMulti(queue &q, float (*matrix_a)[a_columns], float (*matrix_b)[b_columns], 
  float (*matrix_c)[b_columns], float (*matrix_d_parallel)[b_columns]) {

  // Create the range object for the arrays managed by the buffer.
  range<2> num_items{a_rows, b_columns};
  size_t widthA = a_columns;
  size_t widthB = b_columns;
  size_t widthC = b_columns;

  // Create buffers that hold the data shared between the host and the devices.
  // The buffer destructor is responsible to copy the data back to host when it
  // goes out of scope.
  buffer<float, 2> a_buf(reinterpret_cast<float *>(matrix_a), range(a_rows, a_columns));
  buffer<float, 2> b_buf(reinterpret_cast<float *>(matrix_b), range(a_columns, b_columns));
  buffer<float, 2> c_buf(reinterpret_cast<float *>(matrix_c), num_items);
  buffer<float, 2> sum_buf(reinterpret_cast<float *>(matrix_d_parallel), num_items);

#if 0
  // submit a command to write initial data to buffer A on device
  q.submit([&](handler &h) {
    auto a = a_buf.get_access<access::mode::write>(h);
    h.parallel_for(range(a_rows, a_columns), [=](id<2> i) {
      size_t row = i[0];
      size_t col = i[1];
      a[i] = matrix_a[row][col];
    });
  });

  // submit a command to write initial data to buffer B on device
  q.submit([&](handler &h) {
    auto b = b_buf.get_access<access::mode::write>(h);
    h.parallel_for(range(a_columns, b_columns), [=](id<2> i) {
      int row = i[0];
      int col = i[1];
      b[row][col] = matrix_b[row][col];
    });
  });

  // submit a command to write initial data to buffer C on device
  q.submit([&](handler &h) {
    auto c = c_buf.get_access<access::mode::write>(h);
    h.parallel_for(range(a_rows, b_columns), [=](id<2> i) {
      int row = i[0];
      int col = i[1];
      c[row][col] = matrix_c[row][col];
    });
  });
#endif

  // Submit a command group to the queue by a lambda function that contains the
  // data access permission and device computation (kernel).
  q.submit([&](handler &h) {
    // Create an accessor for each buffer with access permission: read, write or
    // read/write. The accessor is a mean to access the memory in the buffer.
    auto a = a_buf.get_access<access::mode::read, access::target::global_buffer>(h);
    auto b = b_buf.get_access<access::mode::read, access::target::global_buffer>(h);
    auto c = c_buf.get_access<access::mode::read, access::target::global_buffer>(h);

    // The sum_accessor is used to store (with write permission) the sum data.
    auto sum = sum_buf.get_access<access::mode::write>(h);
  
    // Use parallel_for to run vector addition in parallel on device. This
    // executes the kernel.
    //    1st parameter is the number of work items.
    //    2nd parameter is the kernel, a lambda that specifies what to do per
    //    work item. The parameter of the lambda is the work item id.
    // DPC++ supports unnamed lambda kernel by default.
    h.parallel_for(num_items, [=](id<2> i) 
      { size_t c_row = i[0], c_col = i[1];

        int s = 0;
        #pragma unroll 2
        for (size_t k = 0; k < widthA; k++)
          s += a[c_row][k] * b[k][c_col]; 

        sum[c_row][c_col]  = c[c_row][c_col] + s;
      });
  });

}

class MM;

void MatrixMulti_st(queue &q, float (*matrix_a)[a_columns], float (*matrix_b)[b_columns], 
  float (*matrix_c)[b_columns], float (*matrix_d_parallel)[b_columns]) {

  // Create the range object for the arrays managed by the buffer.
  range<2> num_items{a_rows, b_columns};
  size_t widthA = a_columns;
  size_t widthB = b_columns;
  size_t widthC = b_columns;

  // Create buffers that hold the data shared between the host and the devices.
  // The buffer destructor is responsible to copy the data back to host when it
  // goes out of scope.
  buffer<float, 2> a_buf(reinterpret_cast<float *>(matrix_a), range(a_rows, a_columns));
  buffer<float, 2> b_buf(reinterpret_cast<float *>(matrix_b), range(a_columns, b_columns));
  buffer<float, 2> c_buf(reinterpret_cast<float *>(matrix_c), num_items);
  buffer<float, 2> sum_buf(reinterpret_cast<float *>(matrix_d_parallel), num_items);

  // Submit a command group to the queue by a lambda function that contains the
  // data access permission and device computation (kernel).
  q.submit([&](handler &h) {
    // Create an accessor for each buffer with access permission: read, write or
    // read/write. The accessor is a mean to access the memory in the buffer.
    auto a = a_buf.get_access<access::mode::read, access::target::global_buffer>(h);
    auto b = b_buf.get_access<access::mode::read, access::target::global_buffer>(h);
    auto c = c_buf.get_access<access::mode::read, access::target::global_buffer>(h);

    // The sum_accessor is used to store (with write permission) the sum data.
    auto sum = sum_buf.get_access<access::mode::write>(h);
  
    // Use parallel_for to run vector addition in parallel on device. This
    // executes the kernel.
    //    1st parameter is the number of work items.
    //    2nd parameter is the kernel, a lambda that specifies what to do per
    //    work item. The parameter of the lambda is the work item id.
    // DPC++ supports unnamed lambda kernel by default.
    h.single_task<MM>([=]() 
      { size_t c_row = num_items[0], c_col = num_items[1];

        int s = 0;
        #pragma unroll 2
        for (size_t k = 0; k < widthA; k++)
          s += a[c_row][k] * b[k][c_col]; 

        sum[c_row][c_col]  = c[c_row][c_col] + s;
      });
  });

}


//************************************
// Demonstrate matrix multiplication both in sequential on CPU and in parallel on device.
//************************************
int main() {
  // Create device selector for the device of your interest.
#if FPGA_EMULATOR
  // DPC++ extension: FPGA emulator selector on systems without FPGA card.
  INTEL::fpga_emulator_selector d_selector;
#elif FPGA
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
  // Intialize values
  for (int i = 0; i < a_rows; i++)
    for (int j = 0; j < b_columns; j++) sum_sequential[i][j] = 0.0;

  float(*sum_parallel)[b_columns] = new float[a_rows][b_columns];
  // Intialize values
  for (int i = 0; i < a_rows; i++)
    for (int j = 0; j < b_columns; j++) sum_parallel[i][j] = 0.0;

  Timer t;

  try {
    queue q(d_selector, dpc_common::exception_handler);

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "Matrix A size: " << a_rows << "," << a_columns << std::endl;
    std::cout << "Matrix B size: " << a_columns << "," << b_columns << std::endl;
    std::cout << "Matrices C, D size: " << a_rows << "," 
              << b_columns << std::endl;

    // Matrix multiplication in DPC++
    MatrixMulti_st(q, A, B, C, sum_parallel);
  } catch (exception const &e) {
    std::cout << "An exception is caught for matrix multiplication.\n";
    std::terminate();
  }

  std::cout << t.elapsed().count() << " seconds\n";

  size_t widthA = a_columns;
  size_t widthB = b_columns;
  size_t widthC = b_columns;

  Timer th;

  // Compute the sum of two arrays in sequential for validation.
  std::cout << "computing on host..." << std::endl;
  for (size_t i = 0; i < a_rows; i++)
    for (size_t j = 0; j < b_columns; j++) {
      sum_sequential[i][j] = C[i][j];
      for (size_t k = 0; k < widthA; k++)
        sum_sequential[i][j] += A[i][k] * B[k][j];
    }
  std::cout << th.elapsed().count() << " seconds\n";

  // Verify that the two arrays are equal.
  for (size_t i = 0; i < a_rows; i++)
    for (size_t j = 0; j < b_columns; j++) 
      if(sum_sequential[i][j] != sum_parallel[i][j]) {
        std::cout << "not equal" << std::endl;
        return -1;
      }

  std::cout << "Matrix multiplication successfully completed on device.\n";
  return 0;
}
