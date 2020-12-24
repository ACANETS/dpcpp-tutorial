CXX = dpcpp
CXXFLAGS = -O2 -g -std=c++17
INCDIR = src/Utils

BUFFER_EXE_NAME = image-conv
BUFFER_SOURCES = src/image-conv.cpp src/Utils/utils.c src/Utils/gold.c src/Utils/bmp-utils.c

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
