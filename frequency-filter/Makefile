CXX = dpcpp
CXXFLAGS = -O2 -g -std=c++17

# Generated using the Intel oneAPI MKL Link Line Advisor
MKL_COMPILE_FLAGS=-DMKL_ILP64  -I"src/Utils ${MKLROOT}/include"
MKL_LINKER_FLAGS =-L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl

BUFFER_EXE_NAME = frequency-filter
BUFFER_SOURCES = src/frequency-filter.cpp

all: build_buffers

build:
	$(CXX) $(CXXFLAGS) $(MKL_COMPILE_FLAGS) $(MKL_LINKER_FLAGS) -o $(BUFFER_EXE_NAME) $(BUFFER_SOURCES)

run:
	./$(BUFFER_EXE_NAME)

run_usm:
	./$(USM_EXE_NAME)

clean:
	rm -rf $(BUFFER_EXE_NAME) $(USM_EXE_NAME)
