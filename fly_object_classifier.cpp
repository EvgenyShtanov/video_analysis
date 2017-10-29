#include <map>
#include <string>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
//#include <cufft.h>
#include <cuda.h>
//#include <cutil.h>
#include "fly_object_classifier.h"
#include "c_image_base.h"
#include "sparse_image_processing.h"
#include "grad.h"
#include "bw_image_processing.h"
#include "vector_base.h"
// #include "cv_bw_processing.h"
#include "cv_bw_processing_qt.h"
#include "cuda_def.h"
#include "cuda_base.h"
#include "bits_base.h"
#include "ocv_base.h"
#include "c/c_invmoments.h"
#include "c/c_img_data_types.h"

typedef unsigned long int ulong;

#define EPS 0.001

int cu_fly_object_classifier::get_nearest_templates_by_width (QSize img_size) {
	int num_of_w_fittest = 0;

	for (int i = -NUM_PIXELS_WIDTH_DIFF; i < NUM_PIXELS_WIDTH_DIFF; ++i) {
		num_of_w_fittest += width_map.values (img_size.width () + i).size ();
	}

	return num_of_w_fittest;
}

void cu_fly_object_classifier::confine_templates_by_width (int img_w) {
	work_templates.clear ();

	for (int i = -NUM_PIXELS_WIDTH_DIFF; i <= NUM_PIXELS_WIDTH_DIFF; ++i) {
		if (img_w + i > 0)
			work_templates << width_map.values (img_w + i).toVector ();
	}

	for (int i = 0; i < work_templates.size (); ++i) {
		VectorInt obj = get_object  (work_templates[i]);
		int w = get_template_params (work_templates[i]).w;

		// QImage qimg_decompressed = decompress_sparse_object (obj, w, COMMON_HEIGHT, 255, 255);
		// qimg_decompressed.save (QString ("temp/confine_templates_by_width/%1_decompressed_t.bmp").arg (work_templates[i]));
	}
}

/**
 * Calculates Riemann sum under template indices on image img. It can be contour or filled object.
 * \param[in] img - input rgb32 or indexed8 image
 * \param[in] indices - indices of pixel in array representation of 2D template
 * \param[in] t_width - width of template
 * \return 
 */
long double estimate_integral_under_contour (QImage const& img, QVector<int> indices, int t_width) {
	long double sum = 0;

	for(int i=0; i < indices.size(); ++i) {
		int x = indices[i] % t_width;
		int y = indices[i] / t_width;
		// fprintf_s (stderr, "x = %d, y = %d; ", x, y);
		if(x < img.width() && y < img.height() && x>=0 && y>=0 )
			sum += long double( qGray(img.pixel(x,y)) ) / 512;
	}
	
	return sum;
}

/**
 * Calculates Riemann sum under template's indices on image img. It can be contour or filled object.
 * \param[in] img - input rgb32 or indexed8 image
 * \param[in] indices - indices of pixel in array representation of 2D template
 * \param[in] t_width - width of template
 * \param[in] shift_x - shift of template relative to image in X dimension
 * \param[in] shift_y - shift of template relative to image in Y dimension
 * \return 
 * Used in TOUCH dll
 */
long double estimate_integral_under_contour (QImage const& img, QVector<int> indices, int t_width, int shift_x, int shift_y) {
	long double sum = 0;
	
	for(int i=0; i < indices.size(); ++i) {
		int x = indices[i] % t_width + shift_x;
		int y = indices[i] / t_width + shift_y;
		// fprintf_s (stderr, "x = %d, y = %d; ", x, y);
		if(x < img.width() && y < img.height() && x>=0 && y>=0 )
			sum += long double( qGray(img.pixel(x,y)) ) / 512;
	}
	
	return sum;
}



/**
 * Calculates number of non-zero elements in image under indices of template 
 * \param[in] img - input rgb32 or indexed8 image
 * \param[in] indices - indices of pixel in array representation of 2D template
 * \param[in] t_width - width of template
 * \param[in] shift_x - shift of template relative to image in X dimension
 * \param[in] shift_y - shift of template relative to image in Y dimension
 * Used in FLY dll
 */
unsigned int intersection_area (QImage const& img, QVector<int> indices, int t_width, int shift_x, int shift_y) {
	unsigned int area = 0;

	for(int i=0; i < indices.size(); ++i) {
		int x = indices[i] % t_width + shift_x;
		int y = indices[i] / t_width + shift_y;
		// fprintf_s (stderr, "x = %d, y = %d; ", x, y);
		if(x < img.width() && y < img.height() && x>=0 && y>=0 && qGray(img.pixel(x,y)) > 0 )
			area += 1;
	}
	
	return area;
}

/**
 *	QImage const& img - input BW image, QImage::Format_Indexed8
 *	QVector<int> indices - object's indices in template
 *	int t_width - width of original template image
 *	int shift_x - shift relative to image coord system in X
 *	int shift_y - shift relative to image coord system in Y
 */
long int nikodim_distance (QImage const& img, QVector<int> indices, int t_width, int shift_x, int shift_y) {
	// calc object area in input bw image
	long int Area_object = areaBW(img);
	// calc area of template
	long int Area_template = indices.size();
	// calc area of intersection
	long int Area_inter = intersection_area(img, indices, t_width, shift_x, shift_y);

	// Calc Nikodim metric: Nic = Area_obj + Area_template - Area_of_intersection
	long int NicodimDist = Area_object + Area_template - 2*Area_inter;
	
	return NicodimDist;
}

/**
 * Compute sum of all elements in image and normalize by 512 
 */
long double weighted_full_sum (QImage const& img) {
	long double sum = 0;

	for(int i=0; i < img.height(); ++i) {
		for(int j=0; j < img.width(); ++j) {
			sum += long double( qGray(img.pixel(j,i)) ) / 512;
		}
	}

	return sum;
}

template<typename T>
T max(T a, T b) {
	return (a > b ? a : b);
}

template<typename T>
T min(T a, T b) {
	return (a < b ? a : b);
}

void enhance_rectangle (QRect & rect, int margin, QRect boundaries) {
	int left = std::max(0, rect.x() - margin);
	int right = std::min(boundaries.width(), rect.width() + margin);
	int top = std::max(0, rect.y() - margin);
	int bottom = std::min(boundaries.height(), rect.height() + margin);

	rect.setCoords(left,top,right,bottom);
}

void I_fly_object_classifier::set_default_params () {
	roi_w = 1;
	roi_h = 1; // радиусы области, в которой выполнЯется поиск минимума
	step_rot_X = 2;
	step_rot_Y = 2;
	step_rot_Z = 2;
	step_r = 1;
	n_best_elements = 1;
}

/**
 * Function I_fly_object_classifier::append_template_const_height_hu
 * Compress template to sparse representation, compute some descriptors
 * of template and insert that with other parameters to hash
 */
int I_fly_object_classifier::append_template (QImage template_img, float angleX, float angleY, float angleZ) {
	double tilt;
	template_img = rotate_to_horizontal (template_img, &tilt); // rotate by eigenvalues
	eig_tilts.push_back ((float) tilt);
		
	QRect rect = find_non_zero_rect (template_img);
	template_img = template_img.copy (rect);

	double scale = (double) COMMON_HEIGHT / (double) template_img.height ();
	QImage scaled_t_img = im_scale_c (template_img, scale); /////

	int tid = 0;
	if (! templates_param_hash.empty ())
		tid = templates_param_hash.size (); // id of template

	// debug
	// int Ntpm = 8100; // Num of templates per model
	// int model_n = (int) ceil (float (tid + 1) / Ntpm);
	// fprintf_s (stderr, "model_n = %d, tid = %d dirname = %s\n", model_n, tid, (QString ("D:/projects/TOUCH/fly_run/temp/scaled/%1/%2").arg (scaled_t_img.width ()).arg (model_n)).toStdString ().c_str ());
	// scaled_t_img.save (QString ("D:/projects/TOUCH/fly_run/temp/scaled/t_%1_%2_%3.png").arg (angleX).arg (angleY).arg (angleZ));
	QDir dir (QString ("D:/projects/TOUCH/fly_video_cu_copter/temp/templates/%1").arg (scaled_t_img.width ()));
	if (! dir.exists ()) {
		if (! dir.mkdir (QString ("D:/projects/TOUCH/fly_video_cu_copter/temp/templates/%1").arg (scaled_t_img.width ()))) {
			fprintf_s (stderr, "Can't create dir %s!\n", (QString ("D:/projects/TOUCH/fly_video_cu_copter/temp/templates/%1").arg (scaled_t_img.width ())).toStdString ().c_str ());
			return -1;
		}
	}
	QDir dir2 (QString ("D:/projects/TOUCH/fly_video_cu_copter/temp/templates/%1").arg (scaled_t_img.width ()));
	if (! dir2.exists ()) {
		if (! dir2.mkdir (QString ("D:/projects/TOUCH/fly_video_cu_copter/temp/templates/%1").arg (scaled_t_img.width ()))) {
			fprintf_s (stderr, "Can't create dir %s!\n", (QString ("D:/projects/TOUCH/fly_video_cu_copter/temp/templates/%1").arg (scaled_t_img.width ())).toStdString ().c_str ());
			return -1;
		}
	}
	scaled_t_img.save (QString ("D:/projects/TOUCH/fly_video_cu_copter/temp/templates/%1/t_%3_%4_%5.png").arg (scaled_t_img.width ()).arg (angleX).arg (angleY).arg (angleZ));
	// end debug

	unsigned int cur_compressed_obj_length = 0;
	unsigned int compressed_obj_shift = 0;

	// Compress (from sparse to compact representation) and append compressed template // Store object's pixels only
	vec_object_array.push_back (compress_sparse_image_u (scaled_t_img));

	// Get length or num of pixels in current and previous templates
	cur_compressed_obj_length = vec_object_array.back ().size ();
	if (templates_param_hash.empty ()) {
		compressed_obj_shift = 0;
		tid = 0;
	}
	else {
		compressed_obj_shift = templates_param_hash.value (tid - 1).object_shift + templates_param_hash.value (tid - 1).compressed_obj_length;
	}

	// Insert templates parameters to hash
	templates_param_hash.insert (tid, template_params (tid, angleX, angleY, angleZ, 
		scaled_t_img.width (), cur_compressed_obj_length, compressed_obj_shift));

	width_map.insertMulti (scaled_t_img.width (), tid);

	return 0;
}


cu_fly_object_classifier::cu_fly_object_classifier () {
	set_default_params ();

	// int max_possible_w_fit_sum = 0;
	// int ids_sorted_by_width_CPU = 0;

	cu_img_array = 0; // input image
	cu_img_array_upsd = 0; // input image
	cu_params_array = 0;
	cu_object_array = 0;
	cu_indices_of_fittest = 0; // is being calculated on CPU and sent to GPU with input image array
	cu_energy_table = 0; // size is 
	cu_rotation = 0; // size is 
	cu_energy_table_n_best = 0; // size is n_best_elements
	cu_best_ids = 0;
	input_img_tilt = 0;
	is_memory_allocated = false;
	is_data_copyed = false;
	// is_cpu_data_prepared = false;

	is_upsd = false;
}

cu_fly_object_classifier::~cu_fly_object_classifier () {
	clearMemory();
}

void cu_fly_object_classifier::clearMemory () {
	is_memory_allocated = false;
	is_data_copyed = false;
	input_img_tilt = 0;

	if (cu_img_array)
		cudaFree (cu_img_array);
	if (cu_img_array_upsd)
		cudaFree (cu_img_array_upsd);
	if (cu_params_array)
		cudaFree (cu_params_array);
	if (cu_object_array)
		cudaFree (cu_object_array);
	if (cu_indices_of_fittest)
		cudaFree (cu_indices_of_fittest);
	if (cu_energy_table)
		cudaFree (cu_energy_table);
	if (cu_rotation)
		cudaFree (cu_rotation);
	if (cu_energy_table_n_best)
		cudaFree (cu_energy_table_n_best);
}

int cu_fly_object_classifier::allocate_memory_on_GPU (bool realloc_force) { // allocates required memory on GPU and sets is_memory_allocated to true
	if (is_memory_allocated && realloc_force) {
		fprintf_s (stderr, "realloc_force\n");
		this->clearMemory ();
		is_memory_allocated = false;
		this->allocate_memory_on_GPU ();
	} 
	else if (is_memory_allocated) {
		fprintf_s (stderr, "Memory was already allocated on GPU\n");
		return ErrorAllocateMemoryGPU;
	}
	else {
		cudaError_t mem2d;
		mem2d = cudaMalloc ((void**) &cu_params_array, sizeof (template_params) * templates_param_hash.size ());
		if (mem2d != cudaSuccess) { handle_error (cudaGetErrorString (mem2d), ErrorAllocateMemoryGPU, __FILE__, __LINE__); } ;
		fprintf_s (stderr, "cu_params_array-allocated, size=%d, sizeof(template_params)=%d\n",templates_param_hash.size(),sizeof(template_params));

		if (templates_param_hash.size () != vec_object_array.size ()) {
			handle_error ("allocate_memory_on_GPU_cu: templates_param_hash.size != qv_object_array.size",ErrorAllocateMemoryGPU,__FILE__,__LINE__);
		}
		
		// Get number of all values in each vector in vec_object_array (size of cu_object_array)
		template_params t_temp = templates_param_hash.value (vec_object_array.size () - 1);
		int size_of_obj_array = t_temp.object_shift + t_temp.compressed_obj_length;

		mem2d = cudaMalloc ((void**) &cu_object_array, sizeof (MAPTYPE) * size_of_obj_array);
		if (mem2d != cudaSuccess) { handle_error (cudaGetErrorString (mem2d), ErrorAllocateMemoryGPU, __FILE__, __LINE__); } ;
		fprintf_s (stderr, "cu_object_array - allocated, size = %d\n", size_of_obj_array);

		mem2d = cudaMalloc ((void**) &cu_energy_table_n_best, sizeof (float) * n_best_elements);
		if (mem2d != cudaSuccess) { handle_error (cudaGetErrorString (mem2d), ErrorAllocateMemoryGPU, __FILE__, __LINE__); } ;
		fprintf_s (stderr, "cu_energy_table_n_best - allocated, size = %d\n", n_best_elements);

		mem2d = cudaMalloc ((void**) &cu_best_ids, sizeof (int) * n_best_elements);
		if (mem2d != cudaSuccess) { handle_error (cudaGetErrorString (mem2d), ErrorAllocateMemoryGPU, __FILE__, __LINE__); } ;
		fprintf_s (stderr, "cu_best_ids - allocated, size = %d\n", n_best_elements);
	}
	this->is_memory_allocated = true;
	// fprintf_s (stderr, "is_memory_allocated = true\n");
	return 0;
}

int cu_fly_object_classifier::copy_data_to_GPU (bool recopy_force) {
	if (is_data_copyed && recopy_force) {
		// fprintf_s (stderr, "recopy_force\n");
		this->clearMemory ();
		is_memory_allocated = false;
		this->allocate_memory_on_GPU ();
		this->copy_data_to_GPU ();
	} 
	else if (is_data_copyed) {
		fprintf_s (stderr, "Data was already copied on GPU\n");
		return ErrorCopyMemoryGPU;
	}
	else {
		cudaError_t mem2d;
		template_params *temp_params = 0;
		temp_params = listToArray (templates_param_hash.values ());
		if (! temp_params) { handle_error ("copy_data_to_GPU: temp_params == 0", -1, __FILE__, __LINE__); } ;

		mem2d = cudaMemcpy (cu_params_array, temp_params, sizeof (template_params) * templates_param_hash.size (), cudaMemcpyHostToDevice);
		if(mem2d != cudaSuccess) { handle_error (cudaGetErrorString (mem2d), ErrorCopyMemoryGPU, __FILE__, __LINE__); } ;
		// fprintf_s (stderr, "cu_params_array - copyed, size = %d\n", templates_param_hash.size ());
		delete[] temp_params;

		MAPTYPE *obj_array = 0;
		int obj_array_size;
		obj_array = vecvecToArray (vec_object_array, &obj_array_size); // Uses cudaHostAlloc call
		if (obj_array == 0) { handle_error ("copy_data_to_GPU: obj_array == 0", -1, __FILE__, __LINE__); } ;
		// fprintf_s (stderr, "obj_array_size = %d\n", obj_array_size );

		mem2d = cudaMemcpy (cu_object_array, obj_array, sizeof (MAPTYPE) * obj_array_size, cudaMemcpyHostToDevice);
		if (mem2d != cudaSuccess) { handle_error (cudaGetErrorString (mem2d), ErrorCopyMemoryGPU, __FILE__, __LINE__); } ;
		// fprintf_s (stderr, "cu_object_array - copyed, size = %d\n", obj_array_size);
		if (cudaFreeHost (obj_array) != cudaSuccess) { handle_error ("copy_data_to_GPU: cudaFreeHost (obj_array)", -1, __FILE__, __LINE__);	} ;
	}
	this->is_data_copyed = true;
	// fprintf_s (stderr, "is_data_copyed = true\n");
	return 0;
}

int cu_fly_object_classifier::alloc_data_on_gpu_run (int img_width, int img_height) {
	cudaError_t mem2d;

	mem2d = cudaMalloc ((void**)&cu_indices_of_fittest, sizeof (int) * work_templates.size ());
	if( mem2d != cudaSuccess ) {
		fprintf_s (stderr, "Error:cudaMalloc( (void**)&cu_indices_of_fittest, sizeof(int)*work_templates.size():Error message=%s\n", cudaGetErrorString (mem2d) );
		clearMemory ( ); return -1;
	}

	mem2d = cudaMalloc ((void**)&cu_energy_table, sizeof (float) * (work_templates.size () * (2 * roi_h / step_r) * (2 * roi_w / step_r)));
	if (mem2d != cudaSuccess) {
		fprintf_s (stderr, "Error:cudaMalloc( (void**)&cu_energy_table, sizeof(float)*(work_templates.size()*(2*roi_h/step_r)*(2*roi_w/step_r):Error message=%s\n", cudaGetErrorString (mem2d));
		clearMemory (); return -1;
	}
	// fprintf_s (stderr, "cu_energy_table - allocated, size = %d\n", work_templates.size ()*(2*roi_h/step_r)*(2*roi_w/step_r) );

	mem2d = cudaMalloc ((void**)&cu_rotation, sizeof (int) * (work_templates.size () * (2 * roi_h / step_r) * (2 * roi_w / step_r)));
	if( mem2d != cudaSuccess ) {
		fprintf_s (stderr, "Error:cudaMalloc( (void**)&cu_rotation, sizeof(int)*(work_templates.size()*(2*roi_h/step_r)*(2*roi_w/step_r):Error message=%s\n", cudaGetErrorString (mem2d) );
		clearMemory (); return -1;
	}
	// fprintf_s (stderr, "cu_rotation - allocated, size = %d\n", work_templates.size ()*(2*roi_h/step_r)*(2*roi_w/step_r) );

	mem2d = cudaMalloc ((void**)&cu_img_array, sizeof (IMGTYPE) * (img_width * img_height));
	if( mem2d != cudaSuccess ) {
		fprintf_s (stderr, "Error:cudaMalloc( (void**)&cu_img_array, sizeof(IMGTYPE)*(img_eig.width() * img_eig.height() ):Error message=%s\n", cudaGetErrorString (mem2d) );
		clearMemory (); return -1;
	}
	// fprintf_s (stderr, "cu_img_array - allocated, size = %d\n", (input_scaled_img.width () * input_scaled_img.height () ) );

	mem2d = cudaMalloc ((void**)&cu_img_array_upsd, sizeof (IMGTYPE) * (img_width * img_height));
	if( mem2d != cudaSuccess ) {
		fprintf_s (stderr, "Error:cudaMalloc( (void**)&cu_img_array_upsd, sizeof(IMGTYPE)*(img_eig.width() * img_eig.height() ):Error message=%s\n", cudaGetErrorString (mem2d) );
		clearMemory (); return -1;
	}
	// fprintf_s (stderr, "cu_img_array_upsd - allocated, size = %d\n", (input_scaled_img.width () * input_scaled_img.height () ) );
	return 0;
}

int cu_fly_object_classifier::copy_data_to_gpu_run (QImage const& scaled_img) {
	cudaError_t mem2d;

	mem2d = cudaMemcpy (cu_indices_of_fittest, work_templates.data(), sizeof (int) * work_templates.size (), cudaMemcpyHostToDevice);
	if( mem2d != cudaSuccess ) {
		fprintf_s (stderr, "Error:cudaMemcpy( cu_indices_of_fittest, work_templates.data(), sizeof(int) * work_templates.size(), cudaMemcpyHostToDevice): Error message=%s\n", cudaGetErrorString (mem2d) );
		clearMemory ( ); return -1;      
	} ;

	IMGTYPE *img_array = new IMGTYPE[scaled_img.width () * scaled_img.height ()];
	IMGTYPE *img_array_upsd = new IMGTYPE[scaled_img.width () * scaled_img.height ()];
	if (convert_qimage_to_array (img_array, scaled_img)) return -1;
		if (! img_array) { handle_error ("estimate_similarity: img_array == 0", __FILE__, __LINE__); }
	
	mem2d = cudaMemcpy (cu_img_array, img_array, sizeof (IMGTYPE) * scaled_img.width () * scaled_img.height (), cudaMemcpyHostToDevice);
		if (mem2d != cudaSuccess) { handle_error (cudaGetErrorString (mem2d), __FILE__, __LINE__); };
	
	// img_rotate (img_array_upsd, scaled_img.width (), scaled_img.height (), img_array, scaled_img.width (), scaled_img.height (), M_PI);
	img_rotate_upsd (img_array_upsd, img_array, scaled_img.width (), scaled_img.height ());

	mem2d = cudaMemcpy (cu_img_array_upsd, img_array_upsd, sizeof (IMGTYPE) *  scaled_img.width () *  scaled_img.height (), cudaMemcpyHostToDevice);
		if (mem2d != cudaSuccess) { handle_error (cudaGetErrorString (mem2d), __FILE__, __LINE__); };

	delete[] img_array;
	delete[] img_array_upsd;

	return 0;
}

int cu_fly_object_classifier::estimate_similarity (QImage const& img) { // recognition
	if (img.width () == 0 || img.height () == 0) {
		fprintf_s (stderr, "Warning: cu_fly_object_classifier::estimate_similarity: scaled_img is Empty ()\n");
		return -1;
	}	//	{ return handle_error ("estimate_similarity: empty input image", __FILE__, __LINE__); } ;
	int the_best_id = 0;
	
	// Rotate input image by eigenvalues, fit in the closest rectangle, scale to commomn height
	QImage scaled_img;
	if (evaluate_input_img (scaled_img, img) != 0) {
		fprintf_s (stderr, "Warning: cu_fly_object_classifier::estimate_similarity: scaled_img is Empty ()\n");
		return -1;
	}	//return handle_error ("estimate_similarity: evaluate_input_img failed", __FILE__, __LINE__); };

	// scaled_img.save ("scaled_img.jpg");

	// Wrappers for GPU memory operation to support CPU only interface
	if (! this->is_memory_allocated) {this->allocate_memory_on_GPU ();} 
	if (! this->is_data_copyed) {this->copy_data_to_GPU ();}

	confine_templates_by_width (scaled_img.width ());

	if (! work_templates.isEmpty ()) {
		if (alloc_data_on_gpu_run (scaled_img.width (), scaled_img.height ()) != 0) { return handle_error ("estimate_similarity: alloc_data_on_gpu_run failed", __FILE__, __LINE__); };
		if (copy_data_to_gpu_run (scaled_img) != 0) { return handle_error ("estimate_similarity: copy_data_to_gpu_run failed", __FILE__, __LINE__); };
	
		if (run_estimation (scaled_img) != 0) { return handle_error ("estimate_similarity: run_estimation failed", __FILE__, __LINE__); };
	
		if (copy_data_to_cpu_run () != 0) { return handle_error ("estimate_similarity: copy_data_to_cpu_run failed", __FILE__, __LINE__); };

		// fprintf_s (stderr, "The result is %f\n", energy_table[0] );
		int the_index = best_ids[0] / ((2 * roi_h / step_r) * (2 * roi_w / step_r));
		// fprintf_s (stderr, "The index = %d", the_index);
		the_best_id = work_templates[the_index];
		// fprintf_s (stderr, "The id is %d\n", the_best_id);
		// fprintf_s (stderr, "Angles are: x = %f, y = %f, z = %d\n", templates_param_hash.value (the_best_id).angleX, templates_param_hash.value (the_best_id).angleY, 0); // 180*if_rotated[0] );
	}
	else {
		fprintf_s (stderr, "Warning: cu_fly_object_classifier::estimate_similarity: work_templates.isEmpty ()\n");
		return -1;
	}

	return the_best_id;
}

int cu_fly_object_classifier::evaluate_input_img (QImage & out_scaled_img, QImage const& img) {
	// Rotate input image according to eigenvalues
	QImage img_eig = rotate_to_horizontal (img, &input_img_tilt); // input_img_tilt is member 
	img_eig.save ("img_eig.jpg");
	// Fit in smallest rectangle
	QRect rect = find_non_zero_rect (img_eig);
	if (rect.isEmpty ()) { fprintf_s (stderr, "Error: evaluate_input_img: rect.isEmpty ()\n"); return -1; };	// handle_error ("estimate_similarity: rect.isEmpty", __FILE__, __LINE__); }
	img_eig = img_eig.copy (rect);
	img_eig.save ("img_eig_fit.jpg");
	// img_eig.save ("img_eig_fit.bmp");
	// Usage of handle_error is reason why this is member function
	if (img_eig.width () && img_eig.height () == 0) { fprintf_s (stderr, "Error: evaluate_input_img: img_eig is Empty ()\n"); return -1; };	// return handle_error ("estimate_similarity: empty img_eig image", __FILE__, __LINE__); }

	double scale = (double) COMMON_HEIGHT / (double) img_eig.height ();
	out_scaled_img = im_scale_c (img_eig, scale);

	return 0;
}

int cu_fly_object_classifier::run_estimation (QImage const& input_scaled_img) {
	int NUM_BLOCK_X = 1;
	int NUM_BLOCK_Y = work_templates.size ();
    int NUM_THREAD_X = (2 * roi_h / step_r) * (2 * roi_w / step_r);
	int NUM_THREAD_Y = 1;

	QTime RunTimer;  // Timer
	RunTimer.start () ;

	unsigned int img_obj_area = areaBW (input_scaled_img);
	// fprintf_s (stderr, "img_obj_area = %d\n", img_obj_area);

	estimate_nicodim_metric_wrapper (NUM_BLOCK_X, NUM_BLOCK_Y, NUM_THREAD_X, NUM_THREAD_Y,
		cu_img_array, cu_img_array_upsd, input_scaled_img.width (), input_scaled_img.height (), img_obj_area,
		cu_object_array, cu_params_array, cu_indices_of_fittest, roi_w, roi_h, step_r, 
		cu_energy_table, cu_rotation, cu_energy_table_n_best, cu_best_ids, n_best_elements);

	// fprintf_s (stdout, "Time of run func call = %lf seconds\n", double (RunTimer.elapsed ()) / 1000.) ;

	return 0;
}

int cu_fly_object_classifier::copy_data_to_cpu_run () {
	energy_table.resize (n_best_elements);
	best_ids.resize (n_best_elements);
	if_rotated.resize (n_best_elements);

	fprintf_s (stderr, "n_best_elements = %d\n", n_best_elements);

	cudaError_t mem2d = cudaMemcpy (energy_table.data (), cu_energy_table_n_best, sizeof (float) * n_best_elements, cudaMemcpyDeviceToHost);
	if (mem2d != cudaSuccess) {
		fprintf_s (stderr, "Error:cudaMemcpy (energy_table.data (), cu_energy_table_n_best, sizeof (float) * n_best_elements, cudaMemcpyDeviceToHost): Error message=%s\n", cudaGetErrorString (mem2d));
		clearMemory ( ); return -1;      
	} ;
	mem2d = cudaMemcpy (best_ids.data (), cu_best_ids, sizeof (int) * n_best_elements, cudaMemcpyDeviceToHost);
	if (mem2d != cudaSuccess) {
		fprintf_s (stderr, "Error:cudaMemcpy (best_ids.data (), cu_best_ids, sizeof (int) * n_best_elements, cudaMemcpyDeviceToHost): Error message=%s\n", cudaGetErrorString (mem2d));
		clearMemory ( ); return -1;
	} ;
	mem2d = cudaMemcpy (if_rotated.data (), cu_rotation, sizeof (int) * n_best_elements, cudaMemcpyDeviceToHost);
	if (mem2d != cudaSuccess) {
		fprintf_s (stderr, "Error:cudaMemcpy (if_rotated.data (), cu_rotation, sizeof (int) * n_best_elements, cudaMemcpyDeviceToHost): Error message=%s\n", cudaGetErrorString (mem2d));
		clearMemory ( ); return -1;
	} ;

	return 0;
}



/*
	QDir dir (QString ("D:/projects/TOUCH/fly_video_cu_copter/temp/templates/%1").arg (scaled_t_img.width ()));
	if (! dir.exists ()) {
		if (! dir.mkdir (QString ("D:/projects/TOUCH/fly_video_cu_copter/temp/templates/%1").arg (scaled_t_img.width ()))) {
			fprintf_s (stderr, "Can't create dir %s!\n", (QString ("D:/projects/TOUCH/fly_video_cu_copter/temp/templates/%1").arg (scaled_t_img.width ())).toStdString ().c_str ());
			return -1;
		}
	}
	QDir dir2 (QString ("D:/projects/TOUCH/fly_video_cu_copter/temp/templates/%1/%2").arg (scaled_t_img.width ()).arg (model_n));
	if (! dir2.exists ()) {
		if (! dir2.mkdir (QString ("D:/projects/TOUCH/fly_video_cu_copter/temp/templates/%1/%2").arg (scaled_t_img.width ()).arg (model_n))) {
			fprintf_s (stderr, "Can't create dir %s!\n", (QString ("D:/projects/TOUCH/fly_video_cu_copter/temp/templates/%1/%2").arg (scaled_t_img.width ()).arg (model_n)).toStdString ().c_str ());
			return -1;
		}
	}
	scaled_t_img.save (QString ("D:/projects/TOUCH/fly_video_cu_copter/temp/templates/%1/%2/t_%3_%4_%5.png").arg (scaled_t_img.width ()).arg (model_n).arg (angleX).arg (angleY).arg (angleZ));

*/