# Create a library called "PipeArray" which includes the source file *.c.
# The extension is already found. Any number of sources could be listed here.
file( GLOB PIPEARRAY_LIB_SOURCE_FILES *.hpp)
#message(CHECK_START "Found source files in PipeArray/ - ${PIPEARRAY_LIB_SOURCE_FILES}")
add_library (PipeArray STATIC ${PIPEARRAY_LIB_SOURCE_FILES})

# Make sure the compiler can find include files for our PipeArray library
# when other libraries or executables link to Hello
target_include_directories (PipeArray PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})