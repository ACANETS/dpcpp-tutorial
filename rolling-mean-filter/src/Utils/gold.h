#ifndef __GOLD_H__
#define __GOLD_H__

#ifdef __cplusplus
extern "C" {
#endif

int* convolutionGold(int *image, int rows, int cols, 
   float *filter, int filterWidth);
float* convolutionGoldFloat(float *image, int rows, int cols,
   float *filter, int filterWidth);

int* histogramGold(int *data, int items, int bins);
int* histogramGoldFloat(float *data, int items, int bins);

#ifdef __cplusplus
}
#endif

#endif
