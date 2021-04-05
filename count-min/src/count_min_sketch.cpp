# include <iostream>
# include <cmath>
# include <cstdlib>
# include <ctime>
# include <limits>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "count_min_sketch.hpp"

using namespace std;

int cms_total = 0;

/**
   Class definition for CountMinSketch.
   public operations:
   // overloaded updates
   void update(int item, int c);
   void update(char *item, int c);
   // overloaded estimates
   unsigned int estimate(int item);
   unsigned int estimate(char *item);
**/

void cms_init(int C[NUM_D][NUM_W], int hashes[NUM_D][2])
{
  for (auto i = 0; i < NUM_D; i++) {
    for (auto j = 0; j < NUM_W; j++) {
      C[i][j] = 0;
    }
  }
  // initialize d pairwise independent hashes
  for (auto i = 0; i < NUM_D; i++) {
    hashes[i][0] = int(float(rand())*float(LONG_PRIME)/float(RAND_MAX) + 1);
    hashes[i][1] = int(float(rand())*float(LONG_PRIME)/float(RAND_MAX) + 1);
  }
} 

void cms_init_C(int C[NUM_D][NUM_W])
{
  for (auto i = 0; i < NUM_D; i++) {
    for (auto j = 0; j < NUM_W; j++) {
      C[i][j] = 0;
    }
  }
} 

void cms_init_hashes(int hashes[NUM_D][2], CountMinSketch &cm)
{
    // initialize d pairwise independent hashes
  for (auto i = 0; i < NUM_D; i++) {
    hashes[i][0] = cm.hashes[i][0];
    hashes[i][1] = cm.hashes[i][1];
  }
}
// generates a hash value for a char16
unsigned int cms_hashstr(sycl::char16 str) {
  unsigned int hash = 6211; //5381;
  int c=0;
  while (c < 16 && str[c]) {
    hash = ((hash << 5) + hash) + str[c]; /* hash * 33 + c */
    c++;
  }
  return hash;
}

// countMinSketch update item count 
void cms_update(int C[NUM_D][NUM_W], 
  int hashes[NUM_D][2], sycl::char16 str, int c) {
  //cms_total = cms_total + c;
  unsigned int hashval = 0;
  unsigned int item = cms_hashstr(str);
  for (unsigned int j = 0; j < NUM_D; j++) {
    hashval = (hashes[j][0]*item+hashes[j][1])%NUM_W;
    C[j][hashval] = C[j][hashval] + c;
  }
}

// CountMinSketch estimate item count 
unsigned int cms_estimate(int C[NUM_D][NUM_W], 
  int hashes[NUM_D][2], sycl::char16 str) {
  int minval = numeric_limits<int>::max();
  unsigned int hashval = 0;
  unsigned int item = cms_hashstr(str);
  for (unsigned int j = 0; j < NUM_D; j++) {
    hashval = (hashes[j][0]*item+hashes[j][1])%NUM_W;
    minval = MIN(minval, C[j][hashval]);
  }
  return minval;
}

// CountMinSketch constructor
// ep -> error 0.01 < ep < 1 (the smaller the better)
// gamma -> probability for error (the smaller the better) 0 < gamm < 1
CountMinSketch::CountMinSketch(float ep, float gamm) {
  /*
  if (!(0.0001 <= ep && ep < 1)) {
    cout << "eps = "<<ep << " must be in this range: [0.0001, 1)" << endl;
    exit(EXIT_FAILURE);
  } else if (!(0 < gamm && gamm < 1)) {
    cout << "gamma must be in this range: (0,1)" << endl;
    exit(EXIT_FAILURE);
  }
  eps = ep;
  gamma = gamm;
  w = ceil(exp(1)/eps);
  d = ceil(log(1/gamma));
  */
  w = NUM_W;
  d = NUM_D;
  cout << "on Host: CM round up w = " << NUM_W << "; d = " << NUM_D << endl;

  total = 0;
  // initialize counter array of arrays, C
  C = new int *[d];
  unsigned int i, j;
  for (i = 0; i < d; i++) {
    C[i] = new int[w];
    for (j = 0; j < w; j++) {
      C[i][j] = 0;
    }
  }
  std::cout<<"counter array sizes = " << d*w*sizeof(int) << " bytes" <<std::endl;

  // initialize d pairwise independent hashes
  srand(time(NULL));
  hashes = new int* [d];
  for (i = 0; i < d; i++) {
    hashes[i] = new int[2];
    genajbj(hashes, i);
  }
  std::cout<<"hash table sizes = " << d*2*sizeof(int) << " bytes" <<std::endl;

}

// CountMinSkectch destructor
CountMinSketch::~CountMinSketch() {
  // free array of counters, C
  unsigned int i;
  for (i = 0; i < d; i++) {
    delete[] C[i];
  }
  delete[] C;
  
  // free array of hash values
  for (i = 0; i < d; i++) {
    delete[] hashes[i];
  }
  delete[] hashes;
}

// CountMinSketch totalcount returns the
// total count of all items in the sketch
unsigned int CountMinSketch::totalcount() {
  return total;
}

// countMinSketch update item count (int)
void CountMinSketch::update(unsigned int item, int c) {
  total = total + c;
  unsigned int hashval = 0;
  for (unsigned int j = 0; j < d; j++) {
    hashval = (hashes[j][0]*item+hashes[j][1])%w;
    C[j][hashval] = C[j][hashval] + c;
  }
}

// countMinSketch update item count (string)
/*
void CountMinSketch::update(const char *str, int c) {
  int hashval = hashstr(str);
  update(hashval, c);
}
*/

// countMinSketch update item count (string)
void CountMinSketch::update(sycl::char16 str, int c) {
  unsigned int hashval = hashstr(str);
  if(str[0] == 'm' && str[1] == 'a' && str[2] == 't' && str[3] == 't' && str[4] == 'e' && str[5] == 'r')
    std::cout<<"++ CM update: +"<<c<<" hash value="<<hashval << "\n";
  update(hashval, c);
}

// CountMinSketch estimate item count (int)
unsigned int CountMinSketch::estimate(unsigned int item) {
  int minval = numeric_limits<int>::max();
  unsigned int hashval = 0;
  for (unsigned int j = 0; j < d; j++) {
    hashval = (hashes[j][0]*item+hashes[j][1])%w;
    minval = MIN(minval, C[j][hashval]);
  }
  return minval;
}

// CountMinSketch estimate item count (string)
/*
unsigned int CountMinSketch::estimate(const char *str) {
  int hashval = hashstr(str);
  return estimate(hashval);
}
*/

// CountMinSketch estimate item count (string)
unsigned int CountMinSketch::estimate(sycl::char16 str) {
  int hashval = hashstr(str);
  if(str[0] == 'm' && str[1] == 'a' && str[2] == 't' && str[3] == 't' && str[4] == 'e' && str[5] == 'r')
    std::cout<<"************ matter hash="<<hashval<<" "<<estimate(hashval)<<"\n";
  return estimate(hashval);
}

// generates aj,bj from field Z_p for use in hashing
void CountMinSketch::genajbj(int** hashes, int i) {
  hashes[i][0] = int(float(rand())*float(LONG_PRIME)/float(RAND_MAX) + 1);
  hashes[i][1] = int(float(rand())*float(LONG_PRIME)/float(RAND_MAX) + 1);
}

// generates a hash value for a sting
// same as djb2 hash function
/*
unsigned int CountMinSketch::hashstr(const char *str) {
  unsigned long hash = 5381;
  int c;
  while ((c = *str++)) {
    hash = ((hash << 5) + hash) + c; // hash * 33 + c 
  }
  return hash;
}
*/

// generates a hash value for a char16
unsigned int CountMinSketch::hashstr(sycl::char16 str) {
//  unsigned long hash = 5381;
//  int c=0;
//  while (c < 16) {
//    hash = ((hash << 5) + hash) + str[c]; /* hash * 33 + c */
//    c++;
//  }
//  
//  return hash;
  return cms_hashstr(str);
}

