#ifndef FLY_OBJECT_CLASSIFIER_H_
#define FLY_OBJECT_CLASSIFIER_H_

#include <QImage>
#include <QString>
#include <QTime>
#include <QDir>
#include <stdio.h>
#include <math.h>
#include <QVector>
#include <QHash>
#include <QMap>
#include <QVector3D>
#include <QPair>
#include <iostream>
#include <vector>
#include "touch_export.h"
#include "template_params.h"

typedef std::vector<int> VectorInt;
typedef std::vector<float> VectorFloat;
typedef std::vector<std::vector<int> > VectorVectorInt;

#define NUM_PIXELS_WIDTH_DIFF 1 // Difference between image width and widths of taking into account templates (width confinement)
#define COMMON_HEIGHT 16 // Any input fragments and templates are proportionnaly resized to that height
#define MAX_THREDS_PER_BLOCK 1024

#define IMGTYPE float // float or double (float быстрее на CUDA) // тип данных изображений
// #define IMGTYPE unsigned  char
#define MAPTYPE int

// CUDA functions
extern "C" {
	int estimate_nicodim_metric_wrapper (int NUM_BLOCK_X, int NUM_BLOCK_Y,
										 int NUM_THREAD_X, int NUM_THREAD_Y,
										 IMGTYPE *img, IMGTYPE *img_upsd, int w, int h, unsigned int img_obj_area,
										 MAPTYPE *template_ar, template_params *t_params, int *ids_of_fittests, 
										 int RX, int RY, int step_r, 
										 float *energy_temp, int *if_rotated, 
										 float *out_energy_best, int *best_indices, int num_of_best);

};

/**
 *
 *	Auxilliary functions
 *
 **/


/**
 *
 *
 *
 **/

class TOUCH_DLL I_fly_object_classifier {
public:;
	I_fly_object_classifier () { set_default_params (); };
	void set_default_params ();
	virtual ~I_fly_object_classifier (){};
	virtual int append_template (QImage img, float angleX, float angleY, float angleZ);
	virtual int estimate_similarity (QImage const& img) = 0;
	VectorInt get_object (int id) {return vec_object_array[id];} // Gets compressed representation of template with this id (sequentional number)
	VectorVectorInt get_all_object () {return vec_object_array;}
	QHash<int,template_params>  get_all_params () {return templates_param_hash; };
public:;
	int roi_w, roi_h ; //!< Pадиусы области, в которой производится поиск минимума
	int step_r; //!< Шаг скользящего окна по X, Y
	int step_rot_X;
	int step_rot_Y;
	int step_rot_Z;
	int n_best_elements; //!< Num of best templates to store
protected:;
	QHash<int,template_params> templates_param_hash; //!< id, template_params. Unique key <=> unique value
	VectorVectorInt vec_object_array; //!< Array of compressed objects. Position of object's pixel is current_number*shift_array[current_number]
	VectorFloat eig_tilts; // Temporary option // must be deleted!?
	QMap<int,int> width_map; // Width, templates' ids
};

// GPU realization
/**
 *
 * cu_fly_object_classifier - 
 *
 * Calls order :
 * cu_fly_object_classifier
 * append_template - arbitrary num of calls
 * estimate_similarity - arbitrary num of calls
 *
 **/
class TOUCH_DLL cu_fly_object_classifier : public I_fly_object_classifier {
public:;
	cu_fly_object_classifier ();
	virtual ~cu_fly_object_classifier ();
	virtual int estimate_similarity (QImage const& img); // Recognition. Uses Hu's moments
	template_params get_template_params (int id) { return templates_param_hash.value(id); };
	float get_template_tilt (int id) { return eig_tilts[id]; };
	double get_input_image_tilt () { return input_img_tilt; };
	int get_if_rotated (int index) { return (index < if_rotated.size ()) ? if_rotated [index] : -1; };
	int get_num_of_templates () { return templates_param_hash.size(); };
private:;
	bool is_memory_allocated; // if true you cannot call append_template
	bool is_data_copyed;
	// bool is_cpu_data_prepared;
	void clearMemory ();
private:;
	// Auxilliary functions
	int allocate_memory_on_GPU (bool realloc_force = false); // Allocates required memory on GPU and sets is_memory_allocated to true
	int copy_data_to_GPU (bool recopy_force = false); // Copy data to GPU and sets flag
	int alloc_data_on_gpu_run (int img_width, int img_height); // Allocates all necessary data on GPU during estimation run
	int copy_data_to_gpu_run (QImage const& scaled_img); // Copy all necessary data to GPU during estimation

	int evaluate_input_img (QImage & out_scaled_img, QImage const& img); // Rotate input image by eigenvalues, fit in the closest rectangle, scale to commomn height
	int get_nearest_templates_by_width (QSize img_size);
	void confine_templates_by_width (int img_w);
	int run_estimation (QImage const& input_scaled_img);
	int copy_data_to_cpu_run ();
	// Error treatment functions
	int handle_error (const char* err, char* file = __FILE__, int line = __LINE__) { printf ("Error: cu_fly_object_classifier::%s;\n Filename = %s;\nLine number = %d\n", err, file, line); clearMemory (); return -1; };
	int handle_error (const char* err, int ret_val, char* file = __FILE__, int line = __LINE__) { printf ("Error: cu_fly_object_classifier::%s;\n Filename = %s;\nLine number = %d\n", err, file, line); clearMemory (); return ret_val; };
	// CPU data
	QMap<int,int> first_indices_map;
	// CPU temporary data
	QVector<int> work_templates; // ids of confined by width templates
	double input_img_tilt;
	bool is_upsd;
	// Return values
	QVector<float> energy_table;
	QVector<int> best_ids;
	QVector<int> if_rotated;
	// GPU temp data (used on device only, not translated to CPU)
	// GPU data
	IMGTYPE *cu_img_array; // Input image
	IMGTYPE *cu_img_array_upsd; // Input image turned upside down, rotated by 180 degrees
	template_params *cu_params_array;
	MAPTYPE *cu_object_array;
	int *cu_indices_of_fittest; // Calculated on CPU and sent to GPU with input image array
	float *cu_energy_table; // Size is work_templates.size ()*(2*roi_h/step_r)*(2*roi_w/step_r) )
	int *cu_rotation; // Check if input image was rotated. Size is work_templates.size ()*(2*roi_h/step_r)*(2*roi_w/step_r) )
	float *cu_energy_table_n_best; // Size is n_best_elements
	int *cu_best_ids; // Size is n_best_elements
};


#endif


/*
* allocate_memory_on_GPU - only once (non-mandatory, if skipped it will be called with estimate_similarity)
 * copy_data_to_GPU - only once (non-mandatory, if skipped it will be called with estimate_similarity)
*/