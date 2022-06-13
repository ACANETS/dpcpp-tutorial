CXX = dpcpp
CXXFLAGS = -O2 -g -std=c++17
INCDIR = ../image-conv/src/Utils

BUFFER_EXE_NAME = sf-filter
BUFFER_SOURCES = src/sf-filter.cpp ../image-conv/src/Utils/utils.c ../image-conv/src/Utils/gold.c ../image-conv/src/Utils/bmp-utils.c

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
