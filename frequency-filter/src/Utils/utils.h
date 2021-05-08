#ifndef __UTILS_H__
#define __UTILS_H__

/* OpenCL includes */
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

void check(cl_int);

void printCompilerError(cl_program program, cl_device_id device);

char* readFile(const char *filename);

#endif
