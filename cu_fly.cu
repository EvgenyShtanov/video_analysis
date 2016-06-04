// #include "cufft.h" 
// #include "cuda.h" 
// #include "cuda_runtime.h"
#include "math.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
//#include <cufft.h>
#include <cuda.h>
#include "cuda_def.h"
#include "ehMath.h"
#include "template_params.h"
// #include "cu_img_base.cuh"

#define COMMON_HEIGHT 16
#define IMGTYPE float //float or double (float быстрее на CUDA)//тип данных изображений
// #define IMGTYPE unsigned char
#define MAPTYPE int


__global__ void 
estim_nicodim_metric (IMGTYPE			*img, 
					  IMGTYPE			*img_upsd, int w, int h, 
					  unsigned int		 img_obj_area, 
					  MAPTYPE			*template_ar, 
					  template_params	*t_params, 
					  int				*ids_of_fittests, 
					  int				 RX, int RY,
					  int				 step_r, 
					  float				*energy_temp, 
					  int				*if_rotated);

__global__ void 
estim_nicodim_metric_shared  (IMGTYPE			*img, 
							  IMGTYPE			*img_upsd, int w, int h, 
							  unsigned int		 img_obj_area, 
							  MAPTYPE			*template_ar, 
							  template_params	*t_params, 
							  int				*ids_of_fittests, 
							  int				 RX, int RY,
							  int				 step_r, 
							  float				*energy_temp, 
							  int				*if_rotated);


__device__ unsigned int 
calculate_nicodim_metric (IMGTYPE		*img, int w, int h, 
						  unsigned int	 img_obj_area,
						  MAPTYPE		*template_ar, int t_w, int t_h, 
						  unsigned int	 shift, 
						  unsigned int	 t_area, 
						  int RX, int RY, 
						  int			 step_r, 
						  int			 ix);

__device__ unsigned int 
calculate_nicodim_metric_shared (IMGTYPE		*img, int w, int h, 
						  unsigned int	 img_obj_area,
						  MAPTYPE		*template_ar, int t_w, int t_h, 
						  unsigned int	 t_area, 
						  int RX, int RY, 
						  int			 step_r, 
						  int			 ix);

__device__ void 
img_rotate_full (IMGTYPE	*img_out, 
				 int		*w_out,
				 int		*h_out,
				 IMGTYPE	*img_in, int w_in, int h_in, 
				 float		 angle) ;

__device__ void
scale_nearest_neighbourhood (IMGTYPE *img_out, 
							 IMGTYPE *img_in, int w_in, int h_in, 
							 float	  scale_factor);

// __global__ void img_rotate_full(IMGTYPE *img_out, int *w_out,
	// int *h_out,IMGTYPE *img_in, int w_in, int h_in, float angle) ;

__global__ void 
img_turn_upsd (IMGTYPE *img_out, 
			   IMGTYPE *img_in, int w_in, int h_in) ;

__global__ void 
find_min (float		*min_val, 
		  int		*index_best, 
		  int		*if_rotated, 
		  float		*array, int ar_size);

extern "C" {
	int 
	estimate_nicodim_metric_wrapper(int	NUM_BLOCK_X, int	NUM_BLOCK_Y,
									int	NUM_THREAD_X, int	NUM_THREAD_Y,
									IMGTYPE			*img, 
									IMGTYPE			*img_upsd, int	w, int	 h, 
									unsigned int		 img_obj_area,
									MAPTYPE			*template_ar, 
									template_params	*t_params, 
									int				*ids_of_fittests,
									int				 RX, int RY, 
									int				 step_r, 
									float				*energy_temp, 
									int				*if_rotated, 
									float				*out_energy_best, 
									int				*gpu_best_ids, int				 num_of_best);

	int 
	estimate_nicodim_metric_wrapper_shared (int	NUM_BLOCK_X, int	NUM_BLOCK_Y,
											int	NUM_THREAD_X, int	NUM_THREAD_Y,
											IMGTYPE			*img, 
											IMGTYPE			*img_upsd, int	w, int	 h, 
											unsigned int		 img_obj_area,
											MAPTYPE			*template_ar, 
											template_params	*t_params, 
											int				*ids_of_fittests,
											int				 RX, int RY, 
											int				 step_r, 
											float				*energy_temp, 
											int				*if_rotated, 
											float				*out_energy_best, 
											int				*gpu_best_ids, int				 num_of_best);
};

int 
estimate_nicodim_metric_wrapper(int NUM_BLOCK_X, int NUM_BLOCK_Y,
								int NUM_THREAD_X, int NUM_THREAD_Y,
								IMGTYPE			*img, 
								IMGTYPE			*img_upsd, int w, int h, 
								unsigned int	 img_obj_area,
								MAPTYPE			*template_ar, 
								template_params	*t_params, 
								int				*ids_of_fittests, 
								int				 RX, 
								int				 RY, 
								int				 step_r, 
								float			*energy_temp, 
								int				*gpu_if_rotated,
								float			*out_energy_best, 
								int				*gpu_best_ids, int num_of_best) 
{
	dim3 dimGrid  ( NUM_BLOCK_X , NUM_BLOCK_Y ) ;
	dim3 dimBlock ( NUM_THREAD_X, NUM_THREAD_Y ) ;
#ifdef KERNEL_LAUNCH
	// calc Nicodim metric
	estim_nicodim_metric<<<dimGrid, dimBlock>>> (img, img_upsd, w, h, img_obj_area,
		template_ar, t_params, ids_of_fittests, RX, RY, step_r, energy_temp, gpu_if_rotated);

	cudaError_t cu_sync_er = cudaDeviceSynchronize ();
	if (cu_sync_er != cudaSuccess) {
		printf ("Error: estim_nicodim_metric<<<%d,%d;%d,%d>>>(img,w,h,img_obj_area,template_ar,t_params,ids_of_fittests,out_energy_t)-cu_sync_er: Error message=%s\n", 
			NUM_BLOCK_X, NUM_BLOCK_Y, NUM_THREAD_X, NUM_THREAD_Y, cudaGetErrorString (cu_sync_er)) ;

		return ErrorCudaRun ;
	} ;

	/*img_rotate_full<<<1,1>>>(img,&w,&h,img,w,h,M_PI);
	cudaError_t cu_sync_er = cudaDeviceSynchronize() ;
	if( cu_sync_er != cudaSuccess ) {
		printf("Error: img_rotate_full<<<1,1>>> , cu_sync_er: Error message=%s\n", cudaGetErrorString (cu_sync_er)) ;
		return ErrorCudaRun ;
	} ;*/
	find_min<<<1,1>>> (out_energy_best, gpu_best_ids, gpu_if_rotated, energy_temp, NUM_BLOCK_Y*NUM_THREAD_X);
	cu_sync_er = cudaDeviceSynchronize () ;
	if( cu_sync_er != cudaSuccess ) {
		printf ("Error: find_min<<<1,1>>> , cu_sync_er: Error message=%s\n", cudaGetErrorString (cu_sync_er) ) ;
		return ErrorCudaRun ;
	} ;
	// printf("gpu_best_ids [0] = %d\n", *gpu_best_ids);
	// printf("NUM_BLOCK_Y = %d\n", NUM_BLOCK_Y);
	// printf("ids_of_fittests [gpu_best_ids [0] ] = %d\n", ids_of_fittests [ *gpu_best_ids ]);
	// *gpu_best_ids = ids_of_fittests [ *gpu_best_ids ];

#else

#endif

	return 0;
}

int 
estimate_nicodim_metric_wrapper_shared (int NUM_BLOCK_X, int NUM_BLOCK_Y,
										int NUM_THREAD_X, int NUM_THREAD_Y,
										IMGTYPE			*img, 
										IMGTYPE			*img_upsd, int w, int h, 
										unsigned int	 img_obj_area,
										MAPTYPE			*template_ar, 
										template_params	*t_params, 
										int				*ids_of_fittests, 
										int				 RX, 
										int				 RY, 
										int				 step_r, 
										float			*energy_temp, 
										int				*gpu_if_rotated,
										float			*out_energy_best, 
										int				*gpu_best_ids, int num_of_best) 
{
	dim3 dimGrid  ( NUM_BLOCK_X , NUM_BLOCK_Y ) ;
	dim3 dimBlock ( NUM_THREAD_X, NUM_THREAD_Y ) ;
#ifdef KERNEL_LAUNCH
	// calc Nicodim metric
	estim_nicodim_metric<<<dimGrid, dimBlock>>> (img, img_upsd, w, h, img_obj_area,
		template_ar, t_params, ids_of_fittests, RX, RY, step_r, energy_temp, gpu_if_rotated);

	cudaError_t cu_sync_er = cudaDeviceSynchronize ();
	if (cu_sync_er != cudaSuccess) {
		printf ("Error: estim_nicodim_metric<<<%d,%d;%d,%d>>>(img,w,h,img_obj_area,template_ar,t_params,ids_of_fittests,out_energy_t)-cu_sync_er: Error message=%s\n", 
			NUM_BLOCK_X, NUM_BLOCK_Y, NUM_THREAD_X, NUM_THREAD_Y, cudaGetErrorString (cu_sync_er)) ;

		return ErrorCudaRun ;
	} ;

	/*img_rotate_full<<<1,1>>>(img,&w,&h,img,w,h,M_PI);
	cudaError_t cu_sync_er = cudaDeviceSynchronize() ;
	if( cu_sync_er != cudaSuccess ) {
		printf("Error: img_rotate_full<<<1,1>>> , cu_sync_er: Error message=%s\n", cudaGetErrorString (cu_sync_er)) ;
		return ErrorCudaRun ;
	} ;*/
	find_min<<<1,1>>> (out_energy_best, gpu_best_ids, gpu_if_rotated, energy_temp, NUM_BLOCK_Y*NUM_THREAD_X);
	cu_sync_er = cudaDeviceSynchronize () ;
	if( cu_sync_er != cudaSuccess ) {
		printf ("Error: find_min<<<1,1>>> , cu_sync_er: Error message=%s\n", cudaGetErrorString (cu_sync_er) ) ;
		return ErrorCudaRun ;
	} ;
	// printf("gpu_best_ids [0] = %d\n", *gpu_best_ids);
	// printf("NUM_BLOCK_Y = %d\n", NUM_BLOCK_Y);
	// printf("ids_of_fittests [gpu_best_ids [0] ] = %d\n", ids_of_fittests [ *gpu_best_ids ]);
	// *gpu_best_ids = ids_of_fittests [ *gpu_best_ids ];

#else

#endif

	return 0;
}

__global__ void 
estim_nicodim_metric(IMGTYPE			*img, 
		 			 IMGTYPE			*img_upsd, int w, int h, 
					 unsigned int		 img_obj_area,
 					 MAPTYPE			*template_ar, 
 					 template_params	*t_params, 
					 int				*ids_of_fittests, 
					 int				 RX, int RY, 
					 int				 step_r, 
					 float				*energy_temp, 
					 int				*if_rotated)
{
	//-- Block and thread indices
	int by = blockIdx.y ; // 0 .. work_templates.size()
	int tx = threadIdx.x ; // 0 .. (2*roi_h/step_r)*(2*roi_w/step_r)
	int bx = blockIdx.x ; // 1
	// int DBx = blockDim.x ; // (2*roi_h/step_r)*(2*roi_w/step_r)
	// int ix = bx*DBx + tx ;  // 

	int id = ids_of_fittests [by];
	// calculate metric on img using indices of the fittest and store each result in out_energy_t
	int current_index = by * (2*RX / step_r) * (2*RY / step_r) + tx;
	energy_temp[current_index] = 
		(float) calculate_nicodim_metric (img, w, h, img_obj_area, 
		template_ar, t_params[id].w, t_params[id].h, t_params[id].object_shift, 
		t_params[id].compressed_obj_length, RX, RY, step_r, tx) ; //  / t_params [id].compressed_obj_length;
	
	cudaError_t cu_sync_er = cudaDeviceSynchronize ();
	if( cu_sync_er != cudaSuccess ) {
		printf("Error: energy_temp [by * (2*RX / step_r) * (2*RY / step_r) + tx] = (float) calculate_nicodim_metric: Error message=%s\n", 
			cudaGetErrorString (cu_sync_er) ) ;
	} ;

	/* IMGTYPE* img_t = new IMGTYPE [w*h];
	
	dim3 dimGrid  ( 1, w ) ;
	dim3 dimBlock ( h / 2, 1 ) ;
	img_turn_upsd<<<dimGrid, dimBlock>>> (img_t, img, w, h);
	
	cu_sync_er = cudaDeviceSynchronize ();
	if( cu_sync_er != cudaSuccess ) {
		printf("Error: img_turn_upsd<<<dimGrid, dimBlock>>> (img_t, img, w, h): Error message=%s\n", 
			cudaGetErrorString (cu_sync_er) ) ;
	} ;

	delete[] img_t;*/

	// rotate img by 180 dgree 
	// img_rotate_full(img,&w,&h,img,w,h,M_PI);
	// repeat calculation
	float energy_temp2 = 
		(float)calculate_nicodim_metric (img_upsd, w, h, img_obj_area, 
		template_ar, t_params[id].w, t_params[id].h, t_params[id].object_shift, 
		t_params[id].compressed_obj_length, RX, RY, step_r, tx);
	// replace previous energy value by second calculated if needed
	// energy_temp[by*RX*RY + tx] = energy_temp[by*RX*RY + tx] < energy_temp2 ? energy_temp[by*RX*RY + tx] : energy_temp2;
	if (energy_temp[current_index] < energy_temp2) {
		if_rotated[current_index] = 0;
	}
	else {
		if_rotated[current_index] = 1;
		energy_temp[current_index] = energy_temp2;
	}
}

/**
	w - width of input image have to be no greater than COMMON_HEIGHT*2
*/
__global__ void 
estim_nicodim_metric_shared (IMGTYPE			*img, 
		 					 IMGTYPE			*img_upsd, int w, int h, 
							 unsigned int		 img_obj_area,
 							 MAPTYPE			*template_ar, 
 							 template_params	*t_params, 
							 int				*ids_of_fittests, 
							 int				 RX, int RY, 
							 int				 step_r, 
							 float				*energy_temp, 
							 int				*if_rotated)
{
	//-- Block and thread indices
	int by = blockIdx.y ; // 0 .. work_templates.size()
	int tx = threadIdx.x ; // 0 .. (2*roi_h/step_r)*(2*roi_w/step_r)
	int bx = blockIdx.x ; // 1
	// int DBx = blockDim.x ; // (2*roi_h/step_r)*(2*roi_w/step_r)
	// int ix = bx*DBx + tx ;  // 

	// declare buffer for input image in shared memory
	__shared__ IMGTYPE cu_img_buffer[COMMON_HEIGHT * COMMON_HEIGHT * 4];
	__shared__ MAPTYPE cu_template_buffer[COMMON_HEIGHT * COMMON_HEIGHT * 4];

	// copy data to buffer
	for (int i = 0; i < h; ++i) {
		for (int j=0; j < w; ++j)
		cu_img_buffer[i * COMMON_HEIGHT * 4 + j] = img[i * w + j];
	}
	
	int id = ids_of_fittests [by];
	int t_h = t_params[id].h;
	int t_w = t_params[id].w;
	int t_shift = t_params[id].object_shift;
	// copy data to buffer
	for (int i = 0; i < t_h; ++i) {
		for (int j = 0; j < t_w; ++j)
		cu_template_buffer[i * COMMON_HEIGHT * 4 + j] = template_ar[t_shift + i * t_w + j];
	}

	// synchronisation
	__syncthreads ();


	// calculate metric on img using indices of the fittest and store each result in out_energy_t
	int current_index = by * (2 * RX / step_r) * (2 * RY / step_r) + tx;
	energy_temp[current_index] = 
		(float) calculate_nicodim_metric_shared (cu_img_buffer, w, h, img_obj_area, 
		cu_template_buffer, t_params[id].w, t_params[id].h, 
		t_params[id].compressed_obj_length, RX, RY, step_r, tx) ; //  / t_params [id].compressed_obj_length;
	
	cudaError_t cu_sync_er = cudaDeviceSynchronize ();
	if( cu_sync_er != cudaSuccess ) {
		printf("Error: energy_temp [by * (2*RX / step_r) * (2*RY / step_r) + tx] = (float) calculate_nicodim_metric_shared: Error message=%s\n", 
			cudaGetErrorString (cu_sync_er) ) ;
	} ;

	// copy data to buffer
	for (int i = 0; i < h; ++i) {
		for (int j=0; j < w; ++j)
		cu_img_buffer[i*COMMON_HEIGHT*4 + j] = img_upsd[i*w + j];
	}
	// synchronisation
	__syncthreads ();

	// repeat calculation
	float energy_temp2 = 
		(float)calculate_nicodim_metric_shared (cu_img_buffer, w, h, img_obj_area, 
		cu_template_buffer, t_params[id].w, t_params[id].h, 
		t_params[id].compressed_obj_length, RX, RY, step_r, tx);
	// replace previous energy value by second calculated if needed
	// energy_temp[by*RX*RY + tx] = energy_temp[by*RX*RY + tx] < energy_temp2 ? energy_temp[by*RX*RY + tx] : energy_temp2;

	cu_sync_er = cudaDeviceSynchronize ();
	if( cu_sync_er != cudaSuccess ) {
		printf("Error: energy_temp2 [by * (2*RX / step_r) * (2*RY / step_r) + tx] = (float) calculate_nicodim_metric_shared: Error message=%s\n", 
			cudaGetErrorString (cu_sync_er) ) ;
	} ;

	if (energy_temp[current_index] < energy_temp2) {
		if_rotated[current_index] = 0;
	}
	else {
		if_rotated[current_index] = 1;
		energy_temp[current_index] = energy_temp2;
	}
}

__device__ unsigned int 
calculate_nicodim_metric (IMGTYPE *pImg, int ImgW, int ImgH, unsigned int img_obj_area, MAPTYPE*t_array, int t_w, int t_h, unsigned int shift, unsigned int t_area, int RX, int RY, int step_r, int ix) {
	unsigned int Area_inter = 0;

	int shiftX, shiftY ;

	shiftX = (int) ( (float) ImgW/2 - (float)RX + (ix % (RX*2 / step_r) )*step_r - (float)t_w / 2 );
	shiftY = (int) ( (float) ImgH/2 - (float)RY + (ix / (RX*2 / step_r) )*step_r - (float)t_h / 2 );

	for( int i = 0 ; i < t_area; i++ ) {
		int x = t_array [shift+i]%t_w ; // coords of contours elements in map SC
		int y = t_array [shift+i]/t_w ;
		if( ((shiftY+y)*ImgW+shiftX+x) > 0 && ((shiftY+y)*ImgW+shiftX+x) < ImgW*ImgH && (shiftY+y) < ImgH && shiftX+x < ImgW )
			if( pImg [(shiftY+y)*ImgW+shiftX+x] > 0 )
				Area_inter += 1;
	} ;

	return img_obj_area + t_area - 2*Area_inter;
}

__device__ unsigned int 
calculate_nicodim_metric_shared (IMGTYPE *pImg, int ImgW, int ImgH, unsigned int img_obj_area, MAPTYPE*t_array, int t_w, int t_h, unsigned int t_area, int RX, int RY, int step_r, int ix) {
	unsigned int Area_inter = 0;

	int shiftX, shiftY ;

	shiftX = (int) ( (float) ImgW/2 - (float)RX + (ix % (RX*2 / step_r) )*step_r - (float)t_w / 2 );
	shiftY = (int) ( (float) ImgH/2 - (float)RY + (ix / (RX*2 / step_r) )*step_r - (float)t_h / 2 );

	for( int i = 0 ; i < t_area; i++ ) {
		int x = t_array [i]%t_w ; // coords of contours elements in map SC
		int y = t_array [i]/t_w ;
		if( ((shiftY+y)*COMMON_HEIGHT*4+shiftX+x) > 0 && ((shiftY+y)*COMMON_HEIGHT*4+shiftX+x) < COMMON_HEIGHT*4*ImgH && (shiftY+y) < ImgH && shiftX+x < COMMON_HEIGHT*4 )
			if( pImg [(shiftY + y)*COMMON_HEIGHT*4 + shiftX + x] > 0 )
				Area_inter += 1;
	} ;

	return img_obj_area + t_area - 2*Area_inter;
}

__global__ void 
img_turn_upsd (IMGTYPE *img_out, IMGTYPE *img_in, int w_in, int h_in) {
	int x = blockIdx.y ; // 0 .. h_in
	int y = threadIdx.x ; // 0 .. w_in
	
	if ( (h_in - y - 1)*w_in + (w_in - x - 1) < w_in*h_in && (h_in - y - 1)*w_in + (w_in - x - 1) >= 0 && y*w_in + x >= 0 && y*w_in + x < w_in*h_in )
		img_out [y*w_in + x] = img_in [ (h_in - y - 1)*w_in + (w_in - x - 1) ];
	
	y += h_in / 2;
	
	if ( (h_in - y - 1)*w_in + (w_in - x - 1) < w_in*h_in && (h_in - y - 1)*w_in + (w_in - x - 1) >= 0 && y*w_in + x >= 0 && y*w_in + x < w_in*h_in )
		img_out [y*w_in + x] = img_in [ (h_in - y - 1)*w_in + (w_in - x - 1) ];

	// if ( (h_in - b_y - 1)*w_in + (w_in - t_x - 1) < w_in*h_in && (h_in - b_y - 1)*w_in + (w_in - t_x - 1) >= 0 )
		// img_out [b_y*w_in + t_x] = img_in [ (h_in - b_y - 1)*w_in + (w_in - t_x - 1) ];
}

__device__ void 
img_rotate_full(IMGTYPE *img_out, int *w_out,
	int *h_out,IMGTYPE *img_in, int w_in, int h_in, float angle) {
	angle = angle ;

	// calculating center of original and final image
	int xo_in= ceil (double (w_in) / 2);
	int yo_in = ceil (double (h_in) / 2);
	
	double sina = sin (angle);
	double cosa = cos (angle);

	*w_out = w_in * fabs (cosa) + h_in * fabs (sina);
	*h_out = w_in * fabs (sina) + h_in * fabs (cosa);

	int xo_out = ceil (double (*w_out) / 2);
	int yo_out = ceil (double (*h_out) / 2);

	for (int i = 0; i < *h_out ;++i) {
		for (int j = 0; j < *w_out ;++j) {
			double a = double (j-xo_out);
			double b = double (i-yo_out);
			int x = int (a*cosa - b*sina + double (xo_in) );
			int y = int (a*sina + b*cosa + double (yo_in) );

			if (x < w_in && x >= 0 && y >= 0 && y < h_in)
				img_out [i*(*w_out) + j] = img_in [y*w_in + x];
			else
				img_out [i*(*w_out) + j] = 0;
		}
	}
}

__device__ void
scale_nearest_neighbourhood (IMGTYPE *img_out, IMGTYPE *img_in, int w_in, int h_in, float scale_factor){
	int w_out = round(float(w_in) * scale_factor);
	int h_out = round(float(h_in) * scale_factor);

	int xo_in = w_in / 2;
	int yo_in = h_in / 2;

	int xo_out = w_out / 2;
	int yo_out = h_out / 2;

	for (int i=0; i < h_out; ++i) {
		for (int j=0; j < w_out; ++j) {
			float x = float (j-xo_out) / scale_factor;
			float y = float (i-yo_out) / scale_factor;
			
			img_out [i*w_out+j] = img_in [(int (y)+yo_in)*w_in+int (x)+xo_in];
		}
	}
}

__global__ void 
find_min (float *min_val, int *index, int *if_rotated, float *array, int ar_size) {
	// *min_val = pow (2, sizeof(T) ) / 2;
	*min_val = FLT_MAX;
	
	for (int i = 0; i < ar_size; ++i)
		if (array [i] < *min_val) {
			*min_val = array[i];
			*index = i;
			if_rotated[0] = if_rotated[i];
		}
		// *min_val = array[i] < *min_val ? array[i] : *min_val;
}

// pos - position in 1D array. dx = pos % Rx, dy = pos / Rx
__global__ void 
find_min_and_pos (float *min_val, int *index, int *if_rotated, int *pos, float *array, int ar_size) {
	// *min_val = pow( 2, sizeof(T) ) / 2;
	*min_val = FLT_MAX;
	
	for (int i=0;i<ar_size;++i)
		if (array [i] < *min_val) {
			*min_val = array[i];
			*index = i;
			if_rotated[0] = if_rotated[i];
		}
		// *min_val = array[i] < *min_val ? array[i] : *min_val;
}

/*__device__ float device_calc_mean3_sparse2( IMGTYPE*pImg, MAPTYPE*pContour_sparse, int shift, int pMap_sparse_cur_size, int W, int H, 
    int ImgW, int ImgH, int RX, int RY, int ix,  float alpha)
{
	if(pMap_sparse_cur_size <= 0) {
		return 0;
	}

	float ksi = 0; // sum of each value along contour or inside figure
	// int N = 0 ; // num of elements // length of contour
	int shiftX, shiftY ;

	shiftX = (int)((float)ImgW/2 - (float)RX/2 + (float)(ix%RX)-(float)W/2);
	shiftY = (int)((float)ImgH/2 - (float)RY/2 + (float)(ix/RX)-(float)H/2);
	// for( int i = 0 ; i < pMap_sparse_cur_size; i += 2 ) { // workable
	for( int i = 0 ; i < pMap_sparse_cur_size; i++ ) {
		int x = pContour_sparse[shift+i]%W ; // coords of contours elements in map SC
		int y = pContour_sparse[shift+i]/W ;
		// ksi += pImg[(shiftY+i)*ImgW+shiftX+j] ;
		ksi += (float) pImg[(shiftY+y)*ImgW+shiftX+x] ;
		// N ++ ;
	} ;
	// if(N == 0)
	// N = 1;

	return ksi / (512 * __powf( float(pMap_sparse_cur_size), alpha ) );
	// return float(ksi) / (512 * __powf( float(pMap_sparse_cur_size/2), alpha ) );
}*/

/*
	// calc object area in input bw image
	long int Area_object = areaBW(img);
	// calc area of template
	long int Area_template = indices.size();
	// calc area of intersection
	long int Area_inter = intersection_area(img, indices, t_width, shift_x, shift_y);

	// Calc Nikodim metric: Nic = Area_obj + Area_template - Area_of_intersection
	long int NicodimDist = Area_object + Area_template - 2*Area_inter;
	
	return NicodimDist;
*/