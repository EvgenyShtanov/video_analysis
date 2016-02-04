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
#define TEMPLATE_EPS_SIMILARITY 0.2

#define IMGTYPE float //float or double (float быстрее на CUDA) // тип данных изображений
// #define IMGTYPE unsigned  char
#define MAPTYPE int

float normalize_integral(long double estim, float size, float alpha);

class  abstractProgress {
 public:
   //progress = [0,1]
   //stop = {0,1} = {break,continue}
   virtual void progressEvent(float progress, QString const& message)=0;
   virtual bool get_stop()=0;
   virtual void set_stop(bool st)=0;
   virtual ~abstractProgress( ){ };
};

class progressBar:public abstractProgress {
	private:
		float progress;
		bool stop;
	public:
		progressBar() { progress = 0; stop=false; };
		void progressEvent(float x, QString const& msg) {
		progress=x;
		std::cout<<qPrintable(msg)<<", progress: "<<progress<<std::endl;
		};
		void set_stop(bool st) {stop=st;};
		bool get_stop( ) {return stop ;};
		~progressBar ( ) {progress = 0;};
};

struct template_params {
		int id; // order number
		float angleX;
		float angleY;
		float angleZ;
		int w;
		int h;
		int contour_length;
		int compressed_obj_length;
		template_params() {
		this->contour_length = 0;
	}
	template_params(	int tid, float aX, float aY, float aZ, int tw, int th, int cl = 0, int col = 0) {
		id = tid;
		angleX = aX;
		angleY = aY;
		angleZ = aZ;
		w = tw;
		h = th;
		contour_length = cl;
		compressed_obj_length = col;
	}
};

class TOUCH_DLL I_fly_object_classifier {
public:;
	   I_fly_object_classifier(){};
	virtual ~I_fly_object_classifier()=0{};
	// virtual int append_template(QImage img, int model_id, float angleX, float angleY, float angleZ);
};

// CPU realization
class TOUCH_DLL fly_object_classifier : public I_fly_object_classifier {
public:;
	fly_object_classifier();
	virtual ~fly_object_classifier();
	// int set_params(bool UseIntensity );
	void set_default_params();
	int append_template(QImage img, int model_id, float angleX, float angleY, float angleZ);
	int estimate_similarity(QImage const& img);
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
private:;
	QVector<QImage> templates_vector; // if UseIntensity is true // not compressed objects usage version
	// internal values
	QHash<int,template_params> templates_param_hash; // id, template_params. Unique key <=> unique value

	QVector<QVector<MAPTYPE> > qv_contour_array; // array of compressed  contours. Position of contour's pixel == number*shift_array[number]
	QVector<QVector<MAPTYPE> > qv_object_array; // array of compressed objects. Position of object's pixel == number*shift_array[number]
	// return values
	QHash<int, QPair<int, float>  >  energy_map; // id, rotationZ, energy. Multi hash: Unique key --> multiple values. Deprecated
	QHash< QPair< int, int>, float>  energy_hash_table; // id, rotationZ, energy. Not multi hash: Unique key --> the only value
};

/*class prehistory_angle_analizer {

};*/

// GPU realization
/**
Calls order :
cu_fly_object_classifier
append_template - arbitrary num of calls
allocate_memory_on_GPU - only once
copy_data_to_GPU - only once
estimate_similarity - arbitrary num of calls

*/
class cu_fly_object_classifier : public I_fly_object_classifier{
public:;
	cu_fly_object_classifier();
	virtual ~cu_fly_object_classifier();
	// int set_params(bool UseIntensity );
	void set_default_params();
	int append_template(QImage img, int model_id, float angleX, float angleY, float angleZ) ;
	int allocate_memory_on_GPU(); // allocates required memory on GPU and sets is_memory_allocated to true
	int copy_data_to_GPU(); // copy data to GPU and sets falg
	int estimate_similarity(QImage const& img); // recognition
private:;
	bool is_memory_allocated; // if true you cannot call append_template
	bool is_data_copyed;
	void clearMemory( );
public:; // SETTINGS
	bool UseContour ; //!< At least one of this bool params must be true
	bool UseIntensity ;   //!< флаг: средняя яркость по области самолета максимизируется
	float alphaContour ;  //!< коэффициент влиния контура, по-умолчанию 1.0f
	float alphaIntensity ;//!< коэффициент влиния яркости по области самолета , по-умолчанию 1.0f
	int roi_w, roi_h ;          //!< радиусы области по которой выполним поиск минимума
	int step_r; // шаг скольз€щего окна по X, Y
	int step_rot_X;
	int step_rot_Y;
	int step_rot_Z;
	float template_similarity; // eps
	int n_best_elements; // num of best templates to store // if you needn't store all estimated values

private:;
	// internal params
	// CPU data
	QHash<int,template_params> templates_param_hash; // id, template_params. Unique key <=> unique value
	QVector<QVector<MAPTYPE> > qv_contour_array; // array of compressed  contours. Position of contour's pixel == number*shift_array[number]
	QVector<QVector<MAPTYPE> > qv_object_array; // array of sparse contours. Position of contour == number*shift_array[number]
	// return values
	QVector<float> energy_table;
	// GPU data
	IMGTYPE *cu_img_array; // input image
	IMGTYPE *cu_img_array_upside_down; // input image rotated by 180 degrees ??
	template_params* cu_params_array;
	MAPTYPE* cu_object_array;
	int *cu_indices_of_fittest; // is being calculated on CPU and sent to GPU with input image array
	float *cu_energy_table; // size is n_best_elements
	
};

#endif
