/** 
    Daniel Alabi
    Count-Min Sketch Implementation based on paper by
    Muthukrishnan and Cormode, 2004

    minor mods by Yan Luo
**/

#ifndef __CMS_HPP__
#define __CMS_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

// define some constants
# define LONG_PRIME 32993
# define MIN(a,b)  (a < b ? a : b)

/** CountMinSketch class definition here **/
class CountMinSketch {
  // width, depth 
  unsigned int w,d;
  
  // eps (for error), 0.01 < eps < 1
  // the smaller the better
  //float eps;
  
  // gamma (probability for accuracy), 0 < gamma < 1
  // the bigger the better
  //float gamma;
  
  // aj, bj \in Z_p
  // both elements of fild Z_p used in generation of hash
  // function
  //unsigned int aj, bj;

  // total count so far
  unsigned int total; 

  // array of arrays of counters
  int **C;

  // generate "new" aj,bj
  void genajbj(int **hashes, int i);

public:

  // array of hash values for a particular item 
  // contains two element arrays {aj,bj}
  int **hashes;

  // constructor
  CountMinSketch(float eps, float gamma);
  
  // update item (int) by count c
  void update(unsigned int item, int c);
  // update item (string) by count c
  void update(const char *item, int c);

  // update item (char16) by count c
  void update(sycl::char16 item, int c);

  // estimate count of item i and return count
  unsigned int estimate(unsigned int item);
  unsigned int estimate(const char *item);
  unsigned int estimate(sycl::char16 item);

  // return total count
  unsigned int totalcount();

  // generates a hash value for a string
  // same as djb2 hash function
  unsigned int hashstr(const char *str);
  unsigned int hashstr(sycl::char16 str);

  ~CountMinSketch();
};

// define constants for count-min sketch 
// EPS the error, GAMMA the probability
// the size of the counter array is determined by the EP and GAMMA
// NOTE: these numbers should agree with EPS and GAMMA
//#define EPS 0.001
//#define GAMMA 0.01
//#define NUM_W 2719
//#define NUM_D 5

#define EPS 0.0001
#define GAMMA 0.001
#define NUM_W 65536 //32768  //round up from 27183
#define NUM_D 16 //8 //round up from 7

extern int cms_total;
extern int C[NUM_D][NUM_W];
extern int hashes[NUM_D][2];
void cms_init(int C[NUM_D][NUM_W], int hashes[NUM_D][2]);
void cms_init_C(int C[NUM_D][NUM_W]);
void cms_init_hashes(int hashes[NUM_D][2], class CountMinSketch &cm);

unsigned int cms_hashstr(sycl::char16 str);
SYCL_EXTERNAL void cms_update(int local_mem_C[NUM_D][NUM_W], 
  int local_mem_hashes[NUM_D][2], sycl::char16 str, int c);
unsigned int cms_estimate(int local_mem_C[NUM_D][NUM_W], 
  int local_mem_hashes[NUM_D][2], sycl::char16 str) ;

#endif
