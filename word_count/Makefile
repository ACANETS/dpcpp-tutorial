CXX = dpcpp
CXXFLAGS = -O2 -g -std=c++17
INCDIR = ./

BUFFER_EXE_NAME = word-count
BUFFER_SOURCES = src/word-count.cpp 

all: build_buffers

build_buffers:
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -o $(BUFFER_EXE_NAME) $(BUFFER_SOURCES)

build_usm:
	$(CXX) $(CXXFLAGS) -o $(USM_EXE_NAME) $(USM_SOURCES)

run: 
	./$(BUFFER_EXE_NAME)

run_usm: 
	./$(USM_EXE_NAME)

clean: 
	rm -rf $(BUFFER_EXE_NAME) $(USM_EXE_NAME)
