/*=======================================================================
 *
 * DPC++ Example
 *
 * Matrix Multiplication with DPC++
 *
 * Author: Yan Luo
 *
 * Copyright (c) 2020-
 *
 * MIT License
 *
 * Changes: Double Buffering optimization implementation
 * 	    Changes made by Sam St.Pierre, Feb 2021
 * 	    Changes are marked by "**"
 */

#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <iomanip>				//**

#if FPGA || FPGA_EMULATOR || FPGA_PROFILE
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif

#include "dpc_common.hpp"

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

//**
// kTimes = # of times to execute the kernel. kTimes must be >= 2
constexpr int kTimes = 3;

//**
// Number of iterations through the main loop
constexpr int kNumRuns = 1;

// Matrice shapes for this example
// A: a_rows x a_columns
// B: a_columns x b_columns
// C, Sum: a_rows x b_columns
constexpr size_t a_rows = 800;
constexpr size_t a_columns = 1000;
constexpr size_t b_columns = 2000;

//**
class MMstv3;

//**
void MatrixMulti_st_v3 (std::unique_ptr<queue> &q, buffer<float, 2> &buffer_a, buffer<float, 2> &buffer_b, 
		buffer<float, 2> &buffer_c, buffer<float, 2> &buffer_sum, event &e) { 

	std::cout << "MatrixMultiplication using single_task() v3." << std::endl;

	Timer t;

	size_t widthA = a_columns;
	size_t widthB = b_columns;
	size_t widthC = b_columns;

	//Submit to the queue and execute the kernel
	e = q->submit([&](handler &h) {

		// Get kernel access to the buffers
		accessor accessor_a(buffer_a, h, read_only);
		accessor accessor_b(buffer_b, h, read_only);
		accessor accessor_c(buffer_c, h, read_only);
		accessor accessor_sum(buffer_sum, h, write_only, noinit);

		h.single_task<MMstv3>([=]() [[intel::kernel_args_restrict]] {				
			float s = 0;
			#pragma unroll 4
			for (size_t i = 0; i < a_rows * b_columns; i++) {
				size_t row, col;
				row = i / widthC;
				col = i % widthC;
				#pragma unroll 2
				for (size_t k = 0; k < a_columns; k++) {
					s += accessor_a[row][k] * accessor_b[k][col];
					accessor_sum[row][col] = accessor_c[row][col] + s;
				}
			}
		});
	});

	event update_host_event;
	update_host_event = q->submit([&](handler &h) {
		accessor accessor_sum(buffer_sum, h, read_only);
		h.update_host(accessor_sum);
	});

	std::cout << t.elapsed().count() << " seconds\n";
}


ulong SyclGetExecTimeNs(event e) {
	ulong start_time =
		e.get_profiling_info<info::event_profiling::command_start>();
	ulong end_time =
		e.get_profiling_info<info::event_profiling::command_end>();
	return (end_time - start_time);
}


void GetExecTime(event e, ulong &total_kernel_time_per_slot) {
	// Query profiling data
	total_kernel_time_per_slot += SyclGetExecTimeNs(e);
}


void ProcessInput(buffer<float, 2> &abuf, buffer<float, 2> &bbuf, buffer<float, 2> &cbuf) {
	// Generating matrice data

	host_accessor buf_acc_a(abuf, write_only, noinit);
	host_accessor buf_acc_b(bbuf, write_only, noinit);
	host_accessor buf_acc_c(cbuf, write_only, noinit);

	// Initialize values
	for (int i = 0; i < a_rows; i++)
		for (int j = 0; j < a_columns; j++) buf_acc_a[i][j] = 1.0;

	for (int i = 0; i < a_columns; i++)
		for (int j = 0; j < b_columns; j++) buf_acc_b[i][j] = 2.0;

	for (int i = 0; i < a_rows; i++)
		for (int j = 0; j < b_columns; j++) buf_acc_c[i][j] = 3.0;
}


int main() {

	// Create device selector for the device of your interest
	#if FPGA_EMULATOR
		// DPC++ extension: FPGA emulator selector on systems without FPGA card
		INTEL::fpga_emulator_selector d_selector;
	#elif defined(FPGA) || defined(FPGA_PROFILE)
		// DPC++ extension: FPGA selector on systems with FPGA card
		INTEL::fpga_selector d_selector;
	#else
		// The default device selector will select the most performant device
		default_selector d_selector;
	#endif

	// Query about the platform
	unsigned number = 0;
	auto myPlatforms = platform::get_platforms();
	// Loop through the platforms to poke into
	for (auto &onePlatform : myPlatforms) {
		std::cout << ++number << " found .." << std::endl << "Platform: "
			<< onePlatform.get_info<info::platform::name>() << std::endl;
		// Loop through the devices
		auto myDevices = onePlatform.get_devices();
		for (auto &oneDevice : myDevices) {
			std::cout << "Device: " << oneDevice.get_info<info::device::name>() << std::endl;
		}
	}
	std::cout << std::endl;

	
	// Create matrices with row, column and initial value
	// to store the input and output data
	float(*A)[a_columns] = new float[a_rows][a_columns];
	// Initialize values
	for (int i = 0; i < a_rows; i++)
		for (int j = 0; j < a_columns; j++) A[i][j] = 1.0;

	float(*B)[b_columns] = new float[a_columns][b_columns];
	// Initialize values
	for (int i = 0; i < a_columns; i++)
		for (int j = 0; j < b_columns; j++) B[i][j] = 2.0;

	float(*C)[b_columns] = new float[a_rows][b_columns];
	// Initialize values
	for (int i = 0; i < a_rows; i++)
		for (int j = 0; j < b_columns; j++) C[i][j] = 3.0;

	float(*sum_sequential)[b_columns] = new float[a_rows][b_columns];
	float(*sum_stv3)[b_columns] = new float[a_rows][b_columns];
	// Initialize values
	for (int i = 0; i < a_rows; i++)
		for (int j = 0; j < b_columns; j++) {
			sum_sequential[i][j] = 0.0;
			sum_stv3[i][j] = 0.0;
		}

	#ifndef FPGA_PROFILE
	Timer th;
	// Compute the sum of two arrays in sequential for validation
	std::cout << "Computing on host..." << std::endl;
	for (size_t i = 0; i < a_rows; i++)
		for (size_t j = 0; j < b_columns; j++) {
			sum_sequential[i][j] = C[i][j];
			for (size_t k = 0; k < a_columns; k++)
				sum_sequential[i][j] += A[i][k] * B[k][j];
		}
	std::cout << th.elapsed().count() << " seconds\n\n";
	#endif


	try {
		auto prop_list = property_list{property::queue::enable_profiling()};
		std::unique_ptr<queue> q;
		q.reset(new queue(d_selector, dpc_common::exception_handler, prop_list));

		// Print out the device information used for the kernel code
		std::cout << "Running on device: "
			<< q->get_device().get_info<info::device::name>() << "\n";
		std::cout << "Matrix A size: " << a_rows << "," << a_columns << std::endl;
		std::cout << "Matrix B size: " << a_columns << "," << b_columns << std::endl;
		std::cout << "Matrices C, D size: " << a_rows << "," << b_columns << std::endl;
		std::cout << "\n";

		// Create the range object for the arrays managed by the buffer
		range<2> num_items{a_rows, b_columns};

		// Create a vector to store the matrice buffers
		std::vector<buffer<float, 2>> input_buf_a;
		std::vector<buffer<float, 2>> input_buf_b;
		std::vector<buffer<float, 2>> input_buf_c;
		std::vector<buffer<float, 2>> output_buf;	// Sum buffer

		// Allocate vectors to store the host-side copies of the matrice input data
		for (int i = 0; i < 2; i++) {
			input_buf_a.push_back(buffer<float, 2>(range<2>(a_rows, a_columns)));
			input_buf_b.push_back(buffer<float, 2>(range<2>(a_columns, b_columns)));
			input_buf_c.push_back(buffer<float, 2>(range<2>(a_rows, b_columns)));
			output_buf.push_back(buffer<float, 2>(range<2>(a_rows, b_columns)));
		}

		// SYCL events for each kernel launch
		event sycl_events[2];
		
		// Total execution time of kernels in a given slot (in nanoseconds)
		ulong total_kernel_time_per_slot[2];

		// Total execution time of all kernels
		ulong total_kernel_time = 0;


		/*
		 * Main loop. This loop will run twice to show the performance difference without
		 * and with double buffering
		 */
		for (int i = 0; i < kNumRuns; i++) {
			for (int j = 0; j < 2; j++) {
				// Initialize timers to zero
				total_kernel_time_per_slot[j] = 0;
			}

			switch (i) {
				case 0: {
					std::cout << "*** Beginning execution, without double buffering\n";
					break;
				}
				case 1: {
					std::cout << "*** Beginning execution, with double buffering\n";
					break;
				}
				default: {
					std::cout << "*** Beginning execution\n";
				}
			}

			// Start the timer. This will include the time to process the input data
			// for the first 2 kernel executions
			dpc_common::TimeInterval exec_time;

			// Single buffering
			if (i == 0) {
				for (int j = 0; j < kTimes; j++) {
					// Only print every few iterations, just to limit the prints
					if (j % 10 == 0) {
						std::cout << "Launching kernel #" << j << "\n";
					}
			
					ProcessInput(input_buf_a[0], input_buf_b[0], input_buf_c[0]);
					MatrixMulti_st_v3(q, input_buf_a[0], input_buf_b[0], input_buf_c[0], output_buf[0], sycl_events[0]);
					GetExecTime(sycl_events[0], total_kernel_time_per_slot[0]);
				}
			} //Double Buffering
			else {
				// Process Input for first 2 kernel launches and queue them.
				// Then block on processing the output of the first kernel
				ProcessInput(input_buf_a[0], input_buf_b[0], input_buf_c[0]);
				ProcessInput(input_buf_a[1], input_buf_b[1], input_buf_c[1]);


				std::cout << "Launching kernel #0\n";

				MatrixMulti_st_v3(q, input_buf_a[0], input_buf_b[0], input_buf_c[0], output_buf[0], sycl_events[0]);
				for (int i = 1; i < kTimes; i++) {
					// Only print every few iteration, just to limit the prints
					if (i % 10 == 0) {
						std::cout << "Launching kernel #" << i << "\n";
					}

					// Launch the next kernel
					MatrixMulti_st_v3(q, input_buf_a[i % 2], input_buf_b[i % 2], 
							input_buf_c[i % 2], output_buf[i % 2], sycl_events[i % 2]);

					// Get execution time from previous kernel
					GetExecTime(sycl_events[(i - 1) % 2], total_kernel_time_per_slot[(i - 1) % 2]);
					
					// Generate inputs for the next kernel
					ProcessInput(input_buf_a[(i - 1) % 2], input_buf_b[(i - 1) % 2], input_buf_c[(i - 1) % 2]);
				}
				// Get execution time for final kernel
				GetExecTime(sycl_events[(kTimes - 1) % 2], total_kernel_time_per_slot[(kTimes - 1) % 2]);
			}
			
			// Add up the overall kernel execution time
			total_kernel_time = 0;
			for (int i = 0; i < 2; i++) {
				total_kernel_time += total_kernel_time_per_slot[i];
			}

			// Stop the timer
			double time_span = exec_time.Elapsed();

			std::cout << "\nOverall execution time " << ((i == 0) ? "without" : "with") << " double buffering = "
				<< (unsigned)(time_span * 1000) << " ms\n";
			std::cout << "Total kernel-only execution time " << ((i == 0) ? "without" : "with") << " double buffering = "
				<< (unsigned)(total_kernel_time / 1000000) << " ms\n\n";
			// std::cout Throughput

			//
			#ifndef FPGA_PROFILE
			// Verify that the two arrays are equal
			// FIXME. this needs to be moved to ProcessOutput() to choose the correct output buffer
			//        depending on whether we use single buffer or double buffer
			auto host_acc_d = output_buf[0].get_access<cl::sycl::access::mode::read>();
			for (size_t i = 0; i < a_rows; i++)
				for (size_t j = 0; j < b_columns; j++)
					if ((sum_sequential[i][j] - host_acc_d[i][j]) > 0.0001) {
						std::cout << "NOT EQUAL : " << "i=" << i << " j=" << j << ": "
						 	<< sum_sequential[i][j] << " " << host_acc_d[i][j] << std::endl;
						return -1;
					}
				std::cout << "Matrix Multiplication successfully completed on device\n";
			#endif
			//
		}
	} catch (exception const &e) {
		std::cout << "An exception is caught for matrix multiplication.\n";
		std::terminate();
	}
	return 0;
}
