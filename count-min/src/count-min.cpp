#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <functional>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <fstream>
#include <map>
#include <set>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

#include "single_kernel.hpp"
#include "multi_kernel.hpp"
#include "count_min_sketch.hpp"

using namespace sycl;
using namespace std::chrono;

#if defined(FPGA_EMULATOR)
#define FILE_NAME  "kafka-words.txt"
#else
#define FILE_NAME  "kafka-words-v2.txt"
#endif
// data types and constants
// NOTE: this tutorial assumes you are using a sycl::vec datatype. Therefore, 
// 'Type' can only be changed to a different vector datatype (e.g. int16,
// ulong8, etc...)
using Type = char16;

///////////////////////////////////////////////////////////////////////////////
// forward declaration of the functions in this file
// the function definitions are all below the main() function in this file
template<typename T>
void DoWorkOffload(queue& q, T* in, T* out, size_t total_count,
                   size_t iterations, buffer<int, 2> C_buf, buffer<int,2> hashes_buf);

template<typename T>
void DoWorkSingleKernel(queue& q, T* in, T* out,
                        size_t chunks, size_t chunk_count, size_t total_count,
                        size_t inflight_kernels, size_t iterations, 
                        buffer<int,2> C_buf, buffer<int,2> hashes_buf);

template <typename T>
void DoWorkMultiKernel(queue& q, T* in, T* out,
                       size_t chunks, size_t chunk_count, size_t total_count,
                       size_t inflight_kernels, size_t iterations);

template<typename T>
void PrintPerformanceInfo(std::string print_prefix, size_t count,
                          std::vector<double>& latency_ms,
                          std::vector<double>& process_time_ms);
///////////////////////////////////////////////////////////////////////////////

std::ostream& operator<<(std::ostream& os, const char16 &input)
{
    //for (auto const& i: input) {
    //    os << i << " ";
    //}
    for (auto it = 0; it < 16; it++)
    {
        std::cout << input[it];
    }    
    return os;
}


template<typename T>
void print_queue(T q, class CountMinSketch &cms) {
  while(!q.empty()) {
    std::cout<<q.top() << " " << cms.estimate(q.top()) << std::endl;
    q.pop();
  }
  std::cout<<"\n";
}

template<typename T>
void print_top10_hostCMS(T q, class CountMinSketch &cms) {
  auto i=0;
  while(!q.empty() && i<10) {
    std::cout<<q.top() << " " << cms.estimate(q.top()) << std::endl;
    q.pop();
    i++;
  }
  std::cout<<"\n";
}

template<typename T>
void print_top10_deviceCMS(T q) {
  auto i=0;
  while(!q.empty() && i<10) {
    std::cout<<q.top() << " " << cms_estimate(q.top()) << std::endl;
    q.pop();
    i++;
  }
  std::cout<<"\n";
}

template<typename T>
void print_top10_truecount(T q, std::map<unsigned int, int> &true_count) {
  auto i=0;
  while(!q.empty() && i<10) {
    unsigned int h = cms_hashstr(q.top());
    std::cout<<q.top() << " " << true_count[h] << std::endl;
    q.pop();
    i++;
  }
  std::cout<<"\n";
}

int C[NUM_D][NUM_W];
int hashes[NUM_D][2];

int main(int argc, char* argv[]) {
  // default values
#if defined(FPGA_EMULATOR)
  size_t chunks = 1 << 3;         // 16
  size_t chunk_count = 1 << 8;    // 256
  size_t iterations = 1;
#else
  size_t chunks = 1 << 6;         // 64
  size_t chunk_count = 1 << 9;   // 512
  size_t iterations = 1;
#endif
  std::string inputfile_name = std::string(FILE_NAME);

  // create a CMS on host
  CountMinSketch cm(0.0001, 0.01);
  // initialize counter array
  cms_init_C(C);
  // initilize hash using host CMS' hashes
  cms_init_hashes(hashes, cm);

  // create buffers for device
  buffer<int, 2> C_buf(reinterpret_cast<int *>(C),range(NUM_D, NUM_W));
  buffer<int, 2> hashes_buf(reinterpret_cast<int *>(hashes), range(NUM_D,2));

  // This is the number of kernels we will have in the queue at a single time.
  // If this number is set too low (e.g. 1) then we don't take advantage of
  // fast kernel relaunch (see the README). If this number is set to high,
  // then the first kernel launched finishes before we are done launching all
  // the kernels and therefore throughput is decreased.
  size_t inflight_kernels = 2;

  // parse the command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg(argv[i]);

    if (arg == "--help" || arg == "-h") {
      std::cout << "USAGE: "
                << argv[0] 
                << "[--chunks=<int>] "
                << "[--chunk_count=<int>] "
                << "[--inflight_kernels=<int>] "
                << "[--iterations=<int>]\n";
      return 0;
    } else {
      std::string str_after_equals = arg.substr(arg.find("=") + 1);

      if (arg.find("--chunks=") == 0) {
        chunks = atoi(str_after_equals.c_str());
      } else if (arg.find("--chunk_count=") == 0) {
        chunk_count = atoi(str_after_equals.c_str());
      } else if (arg.find("--inflight_kernels=") == 0) {
        inflight_kernels = atoi(str_after_equals.c_str());
      } else if (arg.find("--iterations=") == 0) {
        iterations = std::max(2, atoi(str_after_equals.c_str()) + 1);
      } else {
        inputfile_name = std::string(argv[i]);
        std::cout << "Use input file: '" << argv[i] << "'\n";
        break; 
      }
    }
  }

  // check the chunks
  if (chunks <= 0) {
    std::cerr << "ERROR: 'chunks' must be greater than 0\n";
    std::terminate();
  }

  // check the chunk size
  if (chunk_count <= 0) {
    std::cerr << "ERROR: 'chunk_count' must be greater than 0\n";
    std::terminate();
  }

  // check inflight_kernels
  if (inflight_kernels <= 0) {
    std::cerr << "ERROR: 'inflight_kernels' must be positive\n";
    std::terminate();
  }

  // check the number of iterations
  if (iterations <= 0) {
    std::cerr << "ERROR: 'iterations' must be positive\n";
    std::terminate();
  }

  // compute the total number of elements
  size_t total_count = chunks * chunk_count;

  std::cout << "# Chunks:             " << chunks << "\n";
  std::cout << "Chunk count:          " << chunk_count << "\n";
  std::cout << "Total count:          " << total_count << "\n";
  std::cout << "Iterations:           " << iterations-1 << "\n";
  std::cout << "\n";

  bool passed = true;

  try {
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
    if (!d.get_info<info::device::usm_host_allocations>()) {
      std::cerr << "ERROR: The selected device does not support USM host"
                << " allocations\n";
      std::terminate();
    }

    // the USM input and output data
    Type *in, *out;
    if ((in = malloc_host<Type>(total_count, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'in'\n";
      std::terminate();
    }
    if ((out = malloc_host<Type>(total_count, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'out'\n";
      std::terminate();
    }

    // read input strings from file.
    std::ifstream infile(inputfile_name);
    std::cout << "reading " << total_count <<" words from "<< FILE_NAME << std::endl;
    std::generate_n(in, total_count, [&infile] { 
      std::string a;
      infile>>a; 
      std::vector<char> b(a.begin(), a.end());
      Type c;
      for(auto k=0; k < 16; k++)
        c[k] = b[k];
      return Type(c);});
    //std::cout<<in[0]<< "**" << in[1] << "**" << std::endl;

    auto cmp_hash = [](Type left, Type right) {
//      unsigned int left_hash = cms_hashstr(left);
//      unsigned int right_hash = cms_hashstr(right);
//      return (left_hash < right_hash);
      for(auto i=0; i < 16; i++)
        if(left[i] < right[i])
          return true;
      return false;  
    };
    // init set of data hash to identify unique words
    std::set<Type, decltype(cmp_hash)> unique_words(cmp_hash);

    // do brute force search and count
    // init map 
    std::map<unsigned int, int> true_count;
    for (size_t i = 0; i < total_count; i++) {
      // init map for brute force search count
      unsigned int item = cms_hashstr(in[i]);
      true_count[item] = 0;
      // populate set of unique words;
      unique_words.insert(in[i]);
    }
    std::cout<<"Total # of Unique Words = "<< unique_words.size() << "\n";

    // run Count-Min sketch on host
    // also collect true_count stats
    for (size_t i = 0; i < total_count; i++) {
      cm.update(in[i], 1);
      // we take advantage of this loop to 
      // do brute force count
      unsigned int item = cms_hashstr(in[i]);
      true_count[item] ++;
    }
    std::cout<<"Total count in CM = "<<cm.totalcount()<<std::endl;

    //std::cout<<"matter "<<cm.estimate("matter") << std::endl;

    // use priority queue to search for top K items
    // 
    // 
    // lambda to compare elements using true count
    // this is needed for creating priority_queue of "Type"
    auto cmp_truecount = [&true_count](Type left, Type right) {
      unsigned int left_hash = cms_hashstr(left);
      unsigned int left_count = true_count[left_hash];
      unsigned int right_hash = cms_hashstr(right);
      unsigned int right_count = true_count[right_hash];
      return (left_count < right_count);
    };
    std::priority_queue<Type, std::vector<Type>, decltype(cmp_truecount)> pq_truecount(cmp_truecount);
    for (std::set<Type, decltype(cmp_hash)>::iterator it=unique_words.begin(); 
      it!=unique_words.end(); ++it) {
      pq_truecount.push(*it);
    }
    print_top10_truecount(pq_truecount, true_count);

    // lambda to compare elements that are in hostside CMS
    // this is needed for creating priority_queue of "Type"
    auto cmp_host_cms = [&cm](Type left, Type right) {
      //std::cout<<"in cmp_host_cms left\n";
      unsigned int left_count = cm.estimate(left);
      unsigned int right_count = cm.estimate(right);
      return (left_count < right_count);
    };
    std::priority_queue<Type, std::vector<Type>, decltype(cmp_host_cms)> pq1(cmp_host_cms);
    for (std::set<Type, decltype(cmp_hash)>::iterator it=unique_words.begin(); 
      it!=unique_words.end(); ++it) {
        //std::cout<<*it<<" "<<cm.estimate(*it)<<"\n";
        pq1.push(*it);
    }
    // query top 10 using CMS on host 
    std::cout<<"Top 10 (CMS On Host):\n";
    print_top10_hostCMS(pq1, cm);
    std::cout<<std::endl;

    // a lambda function to validate the results (compare counters)
    auto validate_results = [&] {
      auto mismatch = 0;
      for (size_t i = 0; i < total_count; i++) {
        unsigned int retval_device = cms_estimate(C, hashes, in[i]);
        unsigned int item = cms_hashstr(in[i]);
        auto comp = (true_count[item] == retval_device);
        if (!comp) {
            std::cerr << "WARNING: Some values do not match due to approximation with CM sketch\n"
                      << "in[" << i << "]:\"" << in[i]
                      << "\" | true_count=" << true_count[item] << "\t"<< "CM on device=" 
                      << retval_device << "\n";
          mismatch ++;
        }
        //else 
      }
      std::cerr<< mismatch << " out of "<<total_count<<" mismatches\n";
      return true;
    };


    // a lambda function to validate the results (compare counters)
    auto validate_results_top10 = [&] {
      auto mismatch = 0;
      // identify top 10 using CM sketch
      Type top10[10];

      /*
      for (size_t i = 0; i < total_count; i++) {
        unsigned int retval_device = cms_estimate(C, hashes, in[i]);
        unsigned int item = cms_hashstr(in[i]);
        auto comp = (true_count[item] == retval_device);
        if (!comp) {
            std::cerr << "WARNING: Some values do not match due to approximation with CM sketch\n"
                      << "in[" << i << "]:\"" << in[i]
                      << "\" | true_count=" << true_count[item] << "\t"<< "CM on device=" 
                      << retval_device << "\n";
          mismatch ++;
        }
        //else 
      }
      */
      std::cerr<< mismatch << " out of "<<total_count<<" mismatches\n";
      return true;
    };


    ////////////////////////////////////////////////////////////////////////////
    // run the offload version, which is NOT optimized for latency at all
    std::cout << "Running the basic offload kernel\n";
    DoWorkOffload(q, in, out, total_count, iterations, C_buf, hashes_buf);

    // validate the results using the lambda
    //passed &= validate_results();

    std::cout << "\n";
    ////////////////////////////////////////////////////////////////////////////

    // initialize counter array
    cms_init_C(C);
    std::cout<<"iteration = "<< iterations<<std::endl;

    ////////////////////////////////////////////////////////////////////////////
    // run the optimized (for latency) version that uses fast kernel relaunch
    // by keeping at most 'inflight_kernels' in the SYCL queue at a time
    std::cout << "Running the latency optimized single-kernel design\n";
    DoWorkSingleKernel(q, in, out, chunks, chunk_count, total_count,
                       inflight_kernels, iterations, C_buf, hashes_buf);

    // validate the results using the lambda
    //passed &= validate_results();

    std::cout << "\n";
    ////////////////////////////////////////////////////////////////////////////

#if 0
    ////////////////////////////////////////////////////////////////////////////
    // run the optimized (for latency) version with multiple kernels that uses
    // fast kernel relaunch by keeping at most 'inflight_kernels' in the SYCL
    // queue at a time
    std::cout << "Running the latency optimized multi-kernel design\n";
    DoWorkMultiKernel(q, in, out, chunks, chunk_count, total_count,
                      inflight_kernels, iterations);

    // validate the results using the lambda
    passed &= validate_results();

    std::cout << "\n";
    ////////////////////////////////////////////////////////////////////////////
#endif

    // free the USM pointers
    sycl::free(in, q);
    sycl::free(out, q);

  } catch (exception const& e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";
    // Most likely the runtime couldn't find FPGA hardware!
    if (e.get_cl_code() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }

  if(passed) {
    std::cout << "PASSED\n";
    return 0;
  } else {
    std::cout << "FAILED\n";
    return 1;
  }
}

// the basic offload kernel version (doesn't care about latency)
template<typename T>
void DoWorkOffload(queue& q, T* in, T* out, size_t total_count,
                   size_t iterations, buffer<int, 2> C_buf, buffer<int,2> hashes_buf) {
  // timing data
  std::vector<double> latency_ms(iterations);
  std::vector<double> process_time_ms(iterations);

  for (size_t i = 0; i < iterations; i++) {
    auto start = high_resolution_clock::now();

    // submit single kernel for entire buffer
    // this function is defined in 'single_kernel.hpp'
    auto e = SubmitSingleWorker(q, in, out, total_count, C_buf, hashes_buf);

    // wait on the kernel to finish
    e.wait();

    auto end = high_resolution_clock::now();

    // compute latency and processing time
    duration<double, std::milli> process_time = end - start;

    // in offload designs, the processing time and latency are identical
    // since the synchronization between the host and device is coarse grain
    // (i.e. the synchronization happens once ALL the data has been processed).
    latency_ms[i] = process_time.count();
    process_time_ms[i] = process_time.count();
  }

  // compute and print timing information
  PrintPerformanceInfo<T>("Offload",
                          total_count, latency_ms, process_time_ms);
}

// The single-kernel version of the design.
// This function optimizes for latency (while maintaining throughput) by
// breaking the computation into 'chunks' and launching kernels for each
// chunk. The synchronization of the kernel ending tells the host that the data
// for the given chunk is ready in the output buffer.
template <typename T>
void DoWorkSingleKernel(queue& q, T* in, T* out,
                        size_t chunks, size_t chunk_count, size_t total_count,
                        size_t inflight_kernels, size_t iterations,
                        buffer<int,2> C_buf, buffer<int,2> hashes_buf) {
  std::vector<double> latency_ms(iterations);
  std::vector<double> process_time_ms(iterations);

  // count the number of chunks for which kernels have been started
  size_t in_chunk = 0;

  // count the number of chunks for which kernels have finished 
  size_t out_chunk = 0;

  // use a queue to track the kernels in flight
  // By queueing multiple kernels before waiting on the oldest to finish
  // (inflight_kernels) we still have kernels in the SYCL queue and ready to
  // launch while we call event.wait() on the oldest kernel in the queue.
  // However, if we set 'inflight_kernels' too high, then the time to launch
  // the first set of kernels will be longer than the time for the first kernel
  // to finish and our latency and throughput will be negatively affected.
  std::queue<event> event_q;

  for (size_t i = 0; i < iterations; i++) {
    // reset the output data to catch any untouched data
    std::fill_n(out, total_count, -1);

    // reset counters
    in_chunk = 0;
    out_chunk = 0;

    // clear the queue
    std::queue<event> clear_q;
    std::swap(event_q, clear_q);

    // latency timers
    high_resolution_clock::time_point first_data_in, first_data_out;

    auto start = high_resolution_clock::now();

    do {
      // if we still have kernels to launch, launch them in here
      if (in_chunk < chunks) {
        // launch the kernel
        size_t chunk_offset = in_chunk*chunk_count; 
        // this function is defined in 'single_kernel.hpp'
        auto e = SubmitSingleWorker(q, in + chunk_offset, out + chunk_offset,
                                    chunk_count, C_buf, hashes_buf);

        // push the kernel event into the queue
        event_q.push(e);

        // if this is the first chunk, track the time
        if (in_chunk == 0) first_data_in = high_resolution_clock::now();
        in_chunk++;
      }

      // wait on the earliest kernel to finish if either condition is met:
      //    1) there are a certain number kernels in flight
      //    2) all of the kernels have been launched
      if ((event_q.size() >= inflight_kernels) || (in_chunk >= chunks)) {
        // pop the earliest kernel event we are waiting on
        auto e = event_q.front();
        event_q.pop();

        // wait on it to finish
        e.wait();

        // track the time if this is the first producer/consumer pair
        if (out_chunk == 0) first_data_out = high_resolution_clock::now();

        // The synchronization of the kernels ending tells us that, at this 
        // point, the first 'out_chunk' chunks are valid on the host.
        // NOTE: This is the point where you would consume the output data
        // at (out + out_chunk*chunk_size).
        out_chunk++;
      }
    } while (out_chunk < chunks);

    auto end = high_resolution_clock::now();

    // compute latency and processing time
    duration<double, std::milli> latency = first_data_out - first_data_in;
    duration<double, std::milli> process_time = end - start;
    latency_ms[i] = latency.count();
    process_time_ms[i] = process_time.count();
  }

  // compute and print timing information
  PrintPerformanceInfo<T>("Single-kernel",
                          total_count, latency_ms, process_time_ms);
}

//
// The multi-kernel version of the design.
// Like the single-kernel version of the design, this design optimizes for 
// latency (while maintaining throughput) by breaking the producing and
// consuming of data into chunks. That is, the main kernel pipeline (K0, 
// K1, and K2 from SubmitMultiKernelWorkers above) are enqueued ONCE but
// the producer and consumer kernels, that feed and consume data to the
// the kernel pipeline, are broken into smaller chunks. The synchronization of
// the producer and consumer kernels (specifically, the consumer kernel)
// signals to the host that a new chunk of data is ready in host memory.
// See the README file for more information on why a producer and consumer
// kernel are created for this design style.
//
// The following is a block diagram of this kernel this function creates:
//
//  in |---| ProducePipe |----| Pipe0 |----| Pipe1 |----| ConsumePipe |---| out
// --->| P |============>| K0 |======>| K1 |======>| K2 |============>| C |---->
//     |---|             |----|       |----|       |----|             |---|
//

// the pipes used to produce/consume data
using ProducePipe = pipe<class ProducePipeClass, Type>;
using ConsumePipe = pipe<class ConsumePipeClass, Type>;

template <typename T>
void DoWorkMultiKernel(queue& q, T* in, T* out,
                        size_t chunks, size_t chunk_count, size_t total_count,
                        size_t inflight_kernels, size_t iterations) {
  // timing data
  std::vector<double> latency_ms(iterations);
  std::vector<double> process_time_ms(iterations);

  // count the number of chunks for which kernels have been started
  size_t in_chunk = 0;

  // count the number of chunks for which kernels have finished 
  size_t out_chunk = 0;

  // use a queue to track the kernels in flight
  std::queue<std::pair<event,event>> event_q;

  for (size_t i = 0; i < iterations; i++) {
    // reset the output data to catch any untouched data
    std::fill_n(out, total_count, -1);

    // reset counters
    in_chunk = 0;
    out_chunk = 0;

    // clear the queue
    std::queue<std::pair<event,event>> clear_q;
    std::swap(event_q, clear_q);

    // latency timers
    high_resolution_clock::time_point first_data_in, first_data_out;

    // launch the worker kernels
    // NOTE: these kernels will process ALL of the data (total_count)
    // while the producer/consumer will be broken into chunks
    // this function is defined in 'multi_kernel.hpp'
    auto events = SubmitMultiKernelWorkers<T,
                                           ProducePipe,
                                           ConsumePipe>(q, total_count);

    auto start = high_resolution_clock::now();

    do {
      // if we still have kernels to launch, launch them in here
      if (in_chunk < chunks) {
        // launch the producer/consumer pair for the next chunk of data
        size_t chunk_offset = in_chunk*chunk_count;

        // these functions are defined in 'multi_kernel.hpp'
        event p_e = SubmitProducer<T, ProducePipe>(q, in + chunk_offset,
                                                   chunk_count);
        event c_e = SubmitConsumer<T, ConsumePipe>(q, out + chunk_offset,
                                                   chunk_count);

        // push the kernel event into the queue
        event_q.push(std::make_pair(p_e, c_e));

        // if this is the first chunk, track the time
        if (in_chunk == 0) first_data_in = high_resolution_clock::now();
        in_chunk++;
      }

      // wait on the oldest kernel to finish if any of these conditions are met:
      //    1) there are a certain number kernels in flight
      //    2) all of the kernels have been launched
      //
      // NOTE: 'inflight_kernels' is now the number of inflight
      // producer/consumer kernel pairs
      if ((event_q.size() >= inflight_kernels) || (in_chunk >= chunks)) {
        // grab the oldest kernel event we are waiting on
        auto event_pair = event_q.front();
        event_q.pop();

        // wait on the producer/consumer kernel pair to finish
        event_pair.first.wait();    // producer
        event_pair.second.wait();   // consumer

        // track the time if this is the first producer/consumer pair
        if (out_chunk == 0) first_data_out = high_resolution_clock::now();

        // at this point the first 'out_chunk' chunks are ready to be
        // processed on the host
        out_chunk++;
      }
    } while(out_chunk < chunks);

    // wait for the worker kernels to finish, which should be done quickly
    // since all producer/consumer pairs are done
    for (auto& e : events) {
      e.wait();
    }

    auto end = high_resolution_clock::now();

    // compute latency and processing time
    duration<double, std::milli> latency = first_data_out - first_data_in;
    duration<double, std::milli> process_time = end - start;
    latency_ms[i] = latency.count();
    process_time_ms[i] = process_time.count();
  }

  // compute and print timing information
  PrintPerformanceInfo<T>("Multi-kernel",
                          total_count, latency_ms, process_time_ms);
}

// a helper function to compute and print the performance info
template<typename T>
void PrintPerformanceInfo(std::string print_prefix, size_t count,
                          std::vector<double>& latency_ms,
                          std::vector<double>& process_time_ms) {
  // compute the input size in MB
  double input_size_megabytes = (sizeof(T) * count) * 1e-6;

  // compute the average latency and processing time
  double iterations = latency_ms.size();
  std::cout<<"iterations = " << iterations <<"\n";
  double avg_latency_ms = std::accumulate(latency_ms.begin(),
                                          latency_ms.end(),
                                          0.0) / iterations;
  double avg_processing_time_ms = std::accumulate(process_time_ms.begin(),
                                                  process_time_ms.end(),
                                                  0.0) / iterations;

  // compute the throughput
  double avg_tp_mb_s = input_size_megabytes / (avg_processing_time_ms * 1e-3);

  // print info
  std::cout << std::fixed << std::setprecision(4);
  std::cout << print_prefix
            << " average latency:           " << avg_latency_ms << " ms\n";
  std::cout << print_prefix
            << " average throughput:        " << avg_tp_mb_s  << " MB/s\n";
}
