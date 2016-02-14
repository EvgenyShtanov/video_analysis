#ifndef FLY_OBJECT_CLASSIFIER_H
#define FLY_OBJECT_CLASSIFIER_H

#include <QImage>
#include <QString>
#include <QTime>
#include <QDir>
#include <stdio.h>
#include <math.h>
#include <QVector>
#include <QHash>
#include <QVector3D>
#include <QPair>
#include <iostream>
#include "touch_export.h"
#include "template_params.h"

#define TEMPLATE_EPS_SIMILARITY 0.2
#define COMMON_HEIGHT 32

#define IMGTYPE float //float or double (float быстрее на CUDA) // тип данных изображений
// #define IMGTYPE unsigned  char
#define MAPTYPE int

extern "C" {
	int estimate_nicodim_metric_wrapper(int NUM_BLOCK_X, int NUM_BLOCK_Y,
                         int NUM_THREAD_X, int NUM_THREAD_Y,
                         IMGTYPE *img, int w, int h, unsigned int img_obj_area,
						 MAPTYPE*template_ar, template_params* t_params, int *ids_of_fittests, 
						 int RX, int RY, int step_r, 
						 float * energy_temp, float * out_energy_best, int * best_indices, int num_of_best);
};

float normalize_integral(long double estim, float size, float alpha);

class TOUCH_DLL I_fly_object_classifier {
public:;
	I_fly_object_classifier(){set_default_params();};
	void set_default_params();
	virtual ~I_fly_object_classifier(){};
	virtual int append_template(QImage img, int model_id, float angleX, float angleY, float angleZ);
	virtual int append_template_const_height(QImage img, int model_id, float angleX, float angleY, float angleZ);
	virtual int estimate_similarity(QImage const& img)=0;
public:; // SETTINGS
	bool UseContour ; //!< At least one of this bool params must be true
	bool UseIntensity ;   //!< флаг: средняя яркость по области самолета максимизируется
	float alphaContour ;  //!< коэффициент влиния контура, по-умолчанию 1.0f
	float alphaIntensity ;//!< коэффициент влиния яркости по области самолета , по-умолчанию 1.0f
	int roi_w, roi_h ;          //!< радиусы области, в которой производитс€ поиск минимума
	int step_r; // шаг скольз€щего окна по X, Y
	int step_rot_X;
	int step_rot_Y;
	int step_rot_Z;
	float template_similarity; // eps
	int n_best_elements; // num of best templates to store // if you needn't store all estimated values
protected:;
	QHash<int,template_params> templates_param_hash; // id, template_params. Unique key <=> unique value
	QVector<QVector<MAPTYPE> > qv_contour_array; // array of compressed  contours. Position of contour's pixel == number*shift_array[number]
	QVector<QVector<MAPTYPE> > qv_object_array; // array of compressed objects. Position of object's pixel == number*shift_array[number]
};

// CPU realization
class TOUCH_DLL fly_object_classifier : public I_fly_object_classifier {
public:;
	fly_object_classifier();
	virtual ~fly_object_classifier();
	// int set_params(bool UseIntensity );
	virtual int estimate_similarity(QImage const& img);
	int estimate_similarity_rx_ry(QImage const & img);
	int estimate_similarity_rx_ry_n_elements(QImage const & img);
	int estimate_similarity_rx_ry_n_elements_eigen_rot(QImage const & img);
	int estimate_similarity_rx_ry_n_elements_eigen_rot_Nicodim_dist(QImage const & img);
	int estimate_similarity_rx_ry_n_elements_eigen_rot_Nicodim_dist_temp(QImage const & img);
	QHash<int, QPair<int, float> > get_energy_map(void) {return energy_map;}; // Deprecated
	QHash< QPair< int, int>, float>  get_energy_hash_table(void){return energy_hash_table;};
	QPair<float,float> get_xy_angles(int id){return qMakePair(templates_param_hash.value(id).angleX, templates_param_hash.value(id).angleY);};
	// QImage get_template(int tid){if(UseIntensity)return templates_vector[tid];};
	QImage get_template(int tid){ return  UseIntensity ? templates_vector[tid] : QImage();};
private:;
	void clearMemory( );
private:;
	// internal values
	QVector<QImage> templates_vector; // if UseIntensity is true // not compressed objects usage version
	// return values
	QHash<int, QPair<int, float>  >  energy_map; // id, rotationZ, energy. Multi hash: Unique key --> multiple values. Deprecated
	QHash< QPair< int, int>, float>  energy_hash_table; // id, rotationZ, energy. Not multi hash: Unique key --> the only value
};

// GPU realization
/**
Calls order :
cu_fly_object_classifier
append_template - arbitrary num of calls
allocate_memory_on_GPU - only once
copy_data_to_GPU - only once
estimate_similarity - arbitrary num of calls

*/
class TOUCH_DLL cu_fly_object_classifier : public I_fly_object_classifier {
public:;
	cu_fly_object_classifier();
	virtual ~cu_fly_object_classifier();
	int allocate_memory_on_GPU(bool realloc_force = false); // allocates required memory on GPU and sets is_memory_allocated to true
	int copy_data_to_GPU(bool recopy_force = false); // copy data to GPU and sets falg
	virtual int estimate_similarity(QImage const& img); // recognition
private:;
	bool is_memory_allocated; // if true you cannot call append_template
	bool is_data_copyed;
	void clearMemory( );
private:;
	// return values
	QVector<float> energy_table;
	QVector<int> best_ids;
	// QHash< QPair< int, int>, float>  energy_hash_table; // id, rotationZ, energy. Not multi hash: Unique key --> the only value
	// GPU data
	IMGTYPE *cu_img_array; // input image
	// IMGTYPE *cu_img_array_upside_down; // input image rotated by 180 degrees ??
	template_params* cu_params_array;
	MAPTYPE* cu_object_array;
	int *cu_indices_of_fittest; // is being calculated on CPU and sent to GPU with input image array
	float *cu_energy_table; // size is n_best_elements
	float *cu_energy_table_n_best; // size is n_best_elements
	int *cu_best_ids; // size is n_best_elements
};

#endif
