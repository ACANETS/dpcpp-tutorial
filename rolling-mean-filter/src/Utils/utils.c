/* System includes */
#include <stdio.h>
#include <stdlib.h>

char* readFile(const char *filename) {

   FILE *fp;
   char *fileData;
   long fileSize;

   /* Open the file */
   fp = fopen(filename, "r");
   if (!fp) {
      printf("Could not open file: %s\n", filename);
      exit(-1);
   }

   /* Determine the file size */
   if (fseek(fp, 0, SEEK_END)) {
      printf("Error reading the file\n");
      exit(-1);
   }
   fileSize = ftell(fp);
   if (fileSize < 0) {
      printf("Error reading the file\n");
      exit(-1);
   }
   if (fseek(fp, 0, SEEK_SET)) {
      printf("Error reading the file\n");
      exit(-1);
   }

   /* Read the contents */
   fileData = (char*)malloc(fileSize + 1);
   if (!fileData) {
      exit(-1);
   }
   if (fread(fileData, fileSize, 1, fp) != 1) {
      printf("Error reading the file\n");
      exit(-1);
   }

   /* Terminate the string */
   fileData[fileSize] = '\0';

   /* Close the file */
   if (fclose(fp)) {
      printf("Error closing the file\n");
      exit(-1);
   }

   return fileData;
}
