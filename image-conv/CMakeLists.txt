set(CMAKE_CXX_COMPILER "dpcpp")
if(WIN32)
    set(CMAKE_C_COMPILER "clang-cl")
endif()

cmake_minimum_required (VERSION 2.8)

project(ImageConv)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# copy image files
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/Images/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Images/)

add_subdirectory (src)
add_subdirectory (src/Utils)
