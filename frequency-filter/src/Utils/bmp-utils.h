#ifndef __BMPFUNCS_H__
#define __BMPFUNCS_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned char uchar;

int* readBmp(const char *filename, int* rows, int* cols);
void writeBmp(int *imageOut, const char *filename, int rows, int cols,
                const char* refFilename);
double* readBmpDouble(const char *filename, int* rows, int* cols, int* depth);
void writeBmpDouble(double *imageOut, const char *filename, int rows, int cols,
                const char* refFilename);

#ifdef __cplusplus
}
#endif

#endif
