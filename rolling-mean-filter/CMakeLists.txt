set(CMAKE_CXX_COMPILER "dpcpp")
if(WIN32)
    set(CMAKE_C_COMPILER "clang-cl")
endif()

cmake_minimum_required (VERSION 2.8)

project(RollingMeanFilter)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# copy image files
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/input/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/input/)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/output/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/output/)

add_subdirectory (src)
add_subdirectory (src/Utils)
