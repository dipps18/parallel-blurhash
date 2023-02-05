#ifndef __BLURHASH_ENCODE_H__
#define __BLURHASH_ENCODE_H__

#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <thrust/reduce.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

const char *blurHashForPixels(int xComponents, int yComponents, int width, int height, uint8_t *rgb, size_t bytesPerRow);

#endif