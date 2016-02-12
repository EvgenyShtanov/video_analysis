#ifndef CU_IMG_BASE_H
#define CU_IMG_BASE_H

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#define IMGTYPE float

__device__ void img_rotate_full(IMGTYPE *img_out, int *w_out,int *h_out,IMGTYPE *img_in, int w_in, int h_in, float angle);


#endif