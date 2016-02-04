#include "cufft.h" 
#include "cuda.h" 
#include "cuda_runtime.h"
#include "stdio.h"
#include "math.h"
// #include "cu_touch.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
//#include <cufft.h>
#include <cuda.h>
#include "cuda_def.h"
#include "ehMath.h"

#define IMGTYPE float //float or double (float быстрее на CUDA)//тип данных изображений
// #define IMGTYPE unsigned char
#define MAPTYPE int


// find n

__global__ void img_rotate_global(IMGTYPE *img_out,IMGTYPE *img_in, int w, int h, float angle);
__device__ void img_rotate(IMGTYPE *img_out,IMGTYPE *img_in, int w, int h, float angle); // keep size
__device__ void img_rotate_full(IMGTYPE *img_out, int *w_out,int *h_out,IMGTYPE *img_in, int w_in, int h_in, float angle);
__device__ void fit_in_closest_frame(IMGTYPE *img_out, int *w_out,int *h_out,IMGTYPE *img_in, int w_in, int h_in);
__device__ void scale_nearest_neighbourhood(IMGTYPE *img_out, IMGTYPE *img_in, int w_in, int h_in, float scale_factor);
__device__ IMGTYPE* horizontal_hist(IMGTYPE*img_in, int w, int h);
__device__ IMGTYPE* vertical_hist(IMGTYPE*img_in, int w, int h);
__device__ IMGTYPE* copy(IMGTYPE*img_in, int w, int h, int xlt, int ylt, int xrb, int yrb); // copy rectangle area

extern "C" {
int img_rotate_wrapper(int NUM_BLOCK_X, int NUM_BLOCK_Y,
                         int NUM_THREAD_X, int NUM_THREAD_Y,
                         IMGTYPE *img, int w, int h, float angle );
};

int img_rotate_wrapper(int NUM_BLOCK_X, int NUM_BLOCK_Y,
                         int NUM_THREAD_X, int NUM_THREAD_Y,
                         IMGTYPE *img_out,IMGTYPE *img_in, int w, int h, float angle )
{
	dim3 dimGrid  ( NUM_BLOCK_X , NUM_BLOCK_Y ) ;
	dim3 dimBlock ( NUM_THREAD_X, NUM_THREAD_Y ) ;

	img_rotate_global<<<dimGrid, dimBlock>>>
			(img_out, img_in, w, h, angle );

	// cudaThreadSynchronize() ; // deprecated
	cudaError_t cu_sync_er = cudaDeviceSynchronize() ;
	if( cu_sync_er != cudaSuccess ) {
		printf("Error:img_rotate_wrapper - cudaDeviceSynchronize: Error message=%s\n", cudaGetErrorString(cu_sync_er) ) ;

		return ErrorCudaRun;
	} ;

	return Success;
}

__global__ void img_rotate_global(IMGTYPE *img_out,IMGTYPE *img_in, int w, int h, float angle) {

}

__device__ void img_rotate( IMGTYPE *img_out, IMGTYPE *img_in, int w, int h, float angle) {
	angle = angle  ;
	// calculating center of original and final image
	int xo=ceil(float(w)/2);
	int yo=ceil(float(h)/2);
	for(int i=0;i<h;++i) {
		for(int j=0;j<w;++j) {
			int x = (i-yo)*cos(angle) + (j-xo)*sin(angle);
			int y = -(i-yo)*sin(angle) + (j-xo)*cos(angle);
			if(x<w && x>=0 && y>=0 && x < h)
				img_out[y*w+x] = img_in[i*w+j];
		}
	}
}

__device__ void img_rotate_full(IMGTYPE *img_out, int *w_out,int *h_out,IMGTYPE *img_in, int w_in, int h_in, float angle) {
	angle = angle ;

	// calculating center of original and final image
	int xo=ceil(float(w_in)/2);
	int yo=ceil(float(h_in)/2);

	int w_full = w_in * cos(angle) + h_in * sin(angle);
	int h_full = w_in * sin(angle) + h_in * cos(angle);

	*w_out = w_full;
	*h_out = h_full;

	for(int i=0;i < h_in;++i) {
		for(int j=0;j < w_in;++j) {
			int x = (i-yo)*cos(angle) + (j-xo)*sin(angle);
			int y = -(i-yo)*sin(angle) + (j-xo)*cos(angle);
			if(x<w_full && x>=0 && y>=0 && x < h_full)
				img_out[y*w_full+x] = img_in[i*w_in+j];
		}
	}

}

__device__ IMGTYPE* horizontal_sum(IMGTYPE*img_in, int w, int h) {

	IMGTYPE * hist = new IMGTYPE(h);

	for(int i=0;i<h;++i) {
		for(int j=0;j<w;++j) {
			hist[i] += img_in[i*w+j];
		}
	}

	return hist;

}

__device__ IMGTYPE* horizontal_num_of_non_zero(IMGTYPE*img_in, int w, int h) {

	IMGTYPE * hist = new IMGTYPE(h);

	for(int i=0;i<h;++i) {
		for(int j=0;j<w;++j) {
			if(img_in[i*w+j])
				hist[i] += 1;
		}
	}

	return hist;

}


__device__ IMGTYPE* vertical_sum(IMGTYPE*img_in, int w, int h) {


	IMGTYPE * hist = new IMGTYPE(w);

	for(int j=0;j<w;++j) {
		for(int i=0;i<h;++i) {
			hist[j] += img_in[i*w+j];
		}
	}

	return hist;

}

__device__ IMGTYPE* vertical_num_of_non_zero(IMGTYPE*img_in, int w, int h) {


	IMGTYPE * hist = new IMGTYPE(w);

	for(int j=0;j<w;++j) {
		for(int i=0;i<h;++i) {
			if(img_in[i*w+j])
				hist[j] += 1;
		}
	}

	return hist;

}
/**
	IMGTYPE*img_in, int w, int h - input image data
	int xlt, int ylt, int xrb, int yrb - left top and right bottom corners
	(0,0) is left top corner of the input image
*/
__device__ IMGTYPE* copy(IMGTYPE*img_in, int w, int h, int xlt, int ylt, int xrb, int yrb) { // copy rectangle area

	int w_out = xrb - xlt;
	int h_out = yrb - ylt;
	IMGTYPE * copy_img = new IMGTYPE(w_out*h_out);

	for(int j=0;j<w_out;++j) {
		for(int i=0;i<h_out;++i) {
			copy_img[i*w_out + j] = img_in[w*(i+ylt)+j+xlt];
		}
	}

	return copy_img;

}

__device__ void scale_nearest_neighbourhood(IMGTYPE *img_out, IMGTYPE *img_in, int w_in, int h_in, float scale_factor){
	int w_out = round(float(w_in) * scale_factor);
	int h_out = round(float(h_in) * scale_factor);

	int xo_in = w_in / 2;
	int yo_in = h_in / 2;

	int xo_out = w_out / 2;
	int yo_out = h_out / 2;

	for(int i=0;i < h_out;++i) {
		for(int j=0;j < w_out;++j) {
			float x = float(j-xo_out) / scale_factor;
			float y = float(i-yo_out) / scale_factor;
			
			img_out[i*w_out+j] = img_in[(int(y)+yo_in)*w_in+int(x)+xo_in];
		}
	}
}


/**
	Fits bw image to minimal frame.
*/
__device__ void fit_in_closest_frame(IMGTYPE *img_out, int *w_out,int *h_out,IMGTYPE *img_in, int w_in, int h_in) { // strongly deprecated

	// calc histograms
	IMGTYPE* h_hist = horizontal_hist(img_in,w_in,h_in);
	IMGTYPE* v_hist = vertical_hist(img_in,w_in,h_in);
	// analyze histograms: find non-zero strips (belts)

	int leftB = 0;
	int topB = 0;
	int rightB = 0;
	int downB = 0; // coords of boundaries
	
	for(int i=0;i < w_in;++i) {
		if(leftB == 0)
			if(v_hist[i] != 0)
				leftB=i;
		else if(v_hist[i] == 0)
				rightB=i;
	}

	for(int i=0;i<h_in;++i) {
		if(topB==0)
			if(h_hist[i]!=0)
				topB=i;
		else if(h_hist[i]==0)
				downB=i;
	}

	// cutter: copy rectangle area which do not contain zero rows or cols
	img_out = copy(img_in, w_in, h_in, leftB, topB, rightB, downB);
}
