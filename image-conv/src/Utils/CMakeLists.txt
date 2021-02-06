# Create a library called "Utils" which includes the source file *.c.
# The extension is already found. Any number of sources could be listed here.
file( GLOB UTILS_LIB_SOURCE_FILES *.c)
#message(CHECK_START "Found source files in Utils/ - ${UTILS_LIB_SOURCE_FILES}")
add_library (Utils STATIC ${UTILS_LIB_SOURCE_FILES})

# Make sure the compiler can find include files for our Utils library
# when other libraries or executables link to Hello
target_include_directories (Utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})