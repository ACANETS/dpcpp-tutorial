#include <stdio.h>
#include <stdlib.h>

#include "gold.h"

int* convolutionGold(int *image, int rows, int cols, float *filter, int filterWidth)
{
   /* Allocate space for the filtered image */
   int *outputImage;
   outputImage = (int*)malloc(rows*cols*sizeof(int));
   if (!outputImage) { exit(-1); }

   /* Compute the filter width (intentionally truncate) */
   int halfFilterWidth = (int)filterWidth/2;

   /* Iterate over the rows of the source image */
   int i; 
   int j;
   int k;
   int l;
   for (i = 0; i < rows; i++) 
   {
      /* Iterate over the columns of the source image */
      for (j = 0; j < cols; j++) 
      {
         /* Reset sum for new source pixel */
         int sum = 0;
         
         /* Apply the filter to the neighborhood */
         for (k = -halfFilterWidth; k <= halfFilterWidth; k++) 
         {
            for (l = -halfFilterWidth; l <= halfFilterWidth; l++)
            {
               /* Indices used to access the image */
               int r = i+k;
               int c = j+l;
               
               /* Handle out-of-bounds locations by clamping to
                * the border pixel */
               r = (r < 0) ? 0 : r;
               c = (c < 0) ? 0 : c;
               r = (r >= rows) ? rows-1 : r;
               c = (c >= cols) ? cols-1 : c;       
               
               sum += image[r*cols+c] *
                      filter[(k+halfFilterWidth)*filterWidth + 
                         (l+halfFilterWidth)];
            }
         }
         
         /* Write the new pixel value */
         outputImage[i*cols+j] = sum;
      }
   }

   return outputImage;
}

float* convolutionGoldFloat(float *image, int rows, int cols, float *filter, int filterWidth)
{
   /* Allocate space for the filtered image */
   float *outputImage;
   outputImage = (float*)malloc(rows*cols*sizeof(float));
   if (!outputImage) { exit(-1); }

   /* Compute the filter width (intentionally truncate) */
   int halfFilterWidth = (int)filterWidth/2;

   /* Iterate over the rows of the source image */
   int i; 
   int j;
   int k;
   int l;
   for (i = 0; i < rows; i++) 
   {
      /* Iterate over the columns of the source image */
      for (j = 0; j < cols; j++) 
      {
         /* Reset sum for new source pixel */
         float sum = 0;
         
         /* Apply the filter to the neighborhood */
         for (k = -halfFilterWidth; k <= halfFilterWidth; k++) 
         {
            for (l = -halfFilterWidth; l <= halfFilterWidth; l++)
            {
               /* Indices used to access the image */
               int r = i+k;
               int c = j+l;
               
               /* Handle out-of-bounds locations by clamping to
                * the border pixel */
               r = (r < 0) ? 0 : r;
               c = (c < 0) ? 0 : c;
               r = (r >= rows) ? rows-1 : r;
               c = (c >= cols) ? cols-1 : c;       
               
               sum += image[r*cols+c] *
                      filter[(k+halfFilterWidth)*filterWidth + 
                         (l+halfFilterWidth)];
            }
         }
         
         /* Write the new pixel value */
         outputImage[i*cols+j] = sum;
      }
   }

   return outputImage;
}

int* histogramGold(int *data, int items, int bins)
{
   int *refHistogram;

   /* Allocate space for the histogram */
	refHistogram = (int*)malloc(bins*sizeof(int));
   if (!refHistogram) { exit(-1); }

   /* Initialize the histogram to zero */
   int i;
   for (i = 0; i < bins; i++) {
      refHistogram[i] = 0;
   }

   /* Compute the histogram */
   for (i = 0; i < items; i++) {
      if (data[i] >= bins) {
         printf("Warning: Histogram data value out-of-bounds\n");
      }
      refHistogram[data[i]]++;
   }

   return refHistogram;
}

int* histogramGoldFloat(float *data, int items, int bins)
{
   int *refHistogram;

   /* Allocate space for the histogram */
	refHistogram = (int*)malloc(bins*sizeof(int));
   if (!refHistogram) { exit(-1); }

   /* Initialize the histogram to zero */
   int i;
   for (i = 0; i < bins; i++) {
      refHistogram[i] = 0;
   }

   /* Compute the histogram */
   for (i = 0; i < items; i++) {
      if ((int)data[i] >= bins) {
         printf("Warning: Histogram data value out-of-bounds\n");
      }
      refHistogram[(int)data[i]]++;
   }

   return refHistogram;
}
