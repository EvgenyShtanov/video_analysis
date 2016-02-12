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

#define EPS 0.001

// compare functions for template_params structure by ratio of width and height
bool compare_wh_ratio(const template_params &a1, const template_params &a2) {
   return float(a1.w) / float(a1.h) < float(a2.w) / float(a2.h);
}

/**
	
	Returns indices of nearest templates
*/
QVector<int> get_nearest_templates( QVector<template_params>  params_vec, QSize img_size, float similarity ) {
	QVector<int> ids;

	for(int i=0; i < params_vec.size(); ++i )
		if( float( params_vec[i].h) * float(img_size.height()) > EPS ) { 
			if( qAbs( float(params_vec[i].w) / float(params_vec[i].h) - float(img_size.width()) / float(img_size.height() ) )  < similarity  )
				ids.push_back(params_vec[i].id);
		}
		else {
			printf("Division by zero!\n");
		}
	return ids;
}

QVector<int> get_nearest_templates( QHash< int, template_params> params_hash, QSize img_size, float similarity ) {
	QVector<int> ids;

	foreach(template_params t_hash, params_hash)
		if( float(t_hash.h) * float(img_size.height() ) > EPS ) {
			// printf("ok\n");
			if( qAbs(float(t_hash.w) / float(t_hash.h) - float(img_size.width()) / float(img_size.height() ) ) < similarity ) {
				ids.push_back(t_hash.id);
			}
		}
		else {
			printf("Division by zero!\n");
		}
	return ids;
}

long double estimate_integral_under_contour(QImage const& img, QVector<int> indices, int t_width) {
	long double sum = 0;
	
	for(int i=0; i < indices.size(); ++i) {
		int x = indices[i] % t_width;
		int y = indices[i] / t_width;
		// printf("x = %d, y = %d; ", x, y);
		if(x < img.width() && y < img.height() && x>=0 && y>=0 )
			sum += long double( qGray(img.pixel(x,y)) ) / 512;
	}
	
	return sum;
}

long double estimate_integral_under_contour(QImage const& img, QVector<int> indices, int t_width, int shift_x, int shift_y) {
	long double sum = 0;
	
	for(int i=0; i < indices.size(); ++i) {
		int x = indices[i] % t_width + shift_x;
		int y = indices[i] / t_width + shift_y;
		// printf("x = %d, y = %d; ", x, y);
		if(x < img.width() && y < img.height() && x>=0 && y>=0 )
			sum += long double( qGray(img.pixel(x,y)) ) / 512;
	}
	
	return sum;
}

/**
*/
unsigned int intersection_area(QImage const& img, QVector<int> indices, int t_width, int shift_x, int shift_y) {
	unsigned int area = 0;

	for(int i=0; i < indices.size(); ++i) {
		int x = indices[i] % t_width + shift_x;
		int y = indices[i] / t_width + shift_y;
		// printf("x = %d, y = %d; ", x, y);
		if(x < img.width() && y < img.height() && x>=0 && y>=0 && qGray(img.pixel(x,y)) > 0 )
			area += 1;
	}
	
	return area;
}

/**
	QImage const& img - input BW image, QImage::Format_Indexed8
	QVector<int> indices - object's indices in template
	int t_width - width of original template image
	int shift_x - shift relative to image coord sys
	int shift_y
*/
long int nikodim_distance(QImage const& img, QVector<int> indices, int t_width, int shift_x, int shift_y) {
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

long double weighted_full_sum(QImage const& img) {
	long double sum = 0;

	for(int i=0; i < img.height(); ++i) {
		for(int j=0; j < img.width(); ++j) {
			sum += long double( qGray(img.pixel(j,i)) ) / 512;
		}
	}

	return sum;
}

float normalize_integral(long double estim, float size, float alpha) {
	return float(estim) / ( powf( float(size), alpha ) );
	//	return ksi / (512 * __powf( float(pMap_sparse_cur_size), alpha ) );
}


template<typename T>
T max(T a, T b) {
	return (a > b ? a : b);
}

template<typename T>
T min(T a, T b) {
	return (a < b ? a : b);
}

void enhance_rectangle(QRect & rect, int margin, QRect boundaries) {
	int left = max(0, rect.x() - margin);
	int right = min(boundaries.width(), rect.width() + margin);
	int top = max(0, rect.y() - margin);
	int bottom = min(boundaries.height(), rect.height() + margin);

	rect.setCoords(left,top,right,bottom);
}

void I_fly_object_classifier::set_default_params(){
	UseIntensity = false;   //!< флаг: средняя яркость по области самолета минимизируется
	UseContour = true;   //!< флаг: средняя яркость по области самолета минимизируется
	alphaContour = (float)0.7;  //!< коэффициент влиния контура, по-умолчанию 1.0f
	alphaIntensity = (float)0.3;//!< коэффициент влиния яркости по области самолета , по-умолчанию 1.0f
	roi_w = 1;
	roi_h = 1;          //!< радиусы области, в которой выполняетс€ поиск минимума
	template_similarity = (float)TEMPLATE_EPS_SIMILARITY;
	// num_anglesX = 90;
	// num_anglesY;
	// num_anglesZ;
}

int I_fly_object_classifier::append_template(QImage img, int model_id, float angleX, float angleY, float angleZ) {
	img = rotate_to_horizontal(img); // rotate by eigenvalues
		// img.save(QString("D:/projects/TOUCH/fly_run/temp/templates_fit_rot_enhance/t_%1_%2_%3_eig.bmp").arg(angleX).arg(angleY).arg(angleZ) );
	QRect rect = find_non_zero_rect(img);
		// enhance_rectangle( rect, 2, img.rect() );
	img = img.copy(rect);
		// img.save(QString("D:/projects/TOUCH/fly_run/temp/templates_fit_rot_enhance/t_%1_%2_%3.bmp").arg(angleX).arg(angleY).arg(angleZ) );
	
	unsigned int cur_contour_length = 0;
	unsigned int cur_compressed_obj_length = 0;
	unsigned int contour_shift = 0;
	unsigned int compressed_obj_shift = 0;
	int tid = templates_param_hash.size(); // id of template
	if(UseContour) {
		// extract contour and compress (from sparse to compact representation) and append compressed contour
		qv_contour_array.append(create_contour_and_compress(img)); // extract and append
			// QImage test_decompress = decompress_sparse_object( qv_contour_array.last(), img.width(), img.height(), 255, 255 );
			// test_decompress.save("test_decompress.bmp");
		cur_contour_length = qv_contour_array.last().size();
		if(templates_param_hash.empty())
			contour_shift = 0;
		else
			contour_shift = templates_param_hash.value(tid-1).contour_shift + templates_param_hash.value(tid-1).contour_length;
	}
	if(UseIntensity) {
		// compress (from sparse to compact representation) and append compressed template // store object only 
		qv_object_array.append(compress_sparse_image(img));
		cur_compressed_obj_length = qv_object_array.last().size();
		if(templates_param_hash.empty())
			compressed_obj_shift = 0;
		else
			compressed_obj_shift = templates_param_hash.value(tid-1).object_shift + templates_param_hash.value(tid-1).compressed_obj_length;
		// templates_vector.push_back(img);
	}
	
	// append to vector of templates
		// templates_param_vector.push_back(template_params(tid, angleX, angleY, angleZ, img.width(), img.height(), contour_length ));
	templates_param_hash.insert(tid, template_params(tid, angleX, angleY, angleZ, img.width(), img.height(), cur_contour_length, cur_compressed_obj_length, contour_shift, compressed_obj_shift ));
		// sort by width/height ratio
		// qSort( templates_param_vector.begin(), templates_param_vector.end(), compare_wh_ratio);
	return 0;
}

fly_object_classifier::fly_object_classifier(){
	set_default_params();
}

fly_object_classifier::~fly_object_classifier(){
	clearMemory( ) ;
}

void fly_object_classifier::clearMemory( ) {

}

int fly_object_classifier::estimate_similarity(QImage const & img) {
	// make bw
	// QImage img_bw = img2bw(img, 115, 1); // the simplest way, change later to blob analysis
	// img_bw.save("d:/temp/temp/img_bw.bmp");
	// resize, fit in closest frame
	// QPoint leftTopPoint = fit_in_closest_frame(img_bw);
	// img_bw.save("d:/temp/temp/fit_in_closest_frame_a10.bmp");
	// calc gradient of original but cutted image
	// QImage img_grad = gradXY_Sobel( img.copy(leftTopPoint.x(), leftTopPoint.y(), img_bw.width(), img_bw.height()));
	QImage img_grad = gradXY_Sobel( img );

	// get_nearest_templates
	// QVector<int> work_templates = get_nearest_templates(templates_param_vector, template_similarity);
	QVector<int> work_templates = get_nearest_templates(templates_param_hash, img_grad.size(), template_similarity);

	// calc contour integral and (or) integral under object
	for(int i = 0; i < work_templates.size(); ++i) {
		if(img_grad.width() == 0) {
			printf("Division by zero - empty gradient image!\n");
			return 0;
		}	
		// printf("templates_param_hash.value( work_templates[i]).w = %d, img_grad.width() = %d\n", templates_param_hash.value( work_templates[i]).w, img_grad.width());
		float scale = float(templates_param_hash.value( work_templates[i]).w) / float (img_grad.width() );
		// printf("scale = %f\n", scale);

		int tid = templates_param_hash.value(work_templates[i]).id;
		// printf("tid = %d\n", tid);

		QImage scaled_img = im_scale_c(img_grad, scale);
		// scaled_img.save("d:/temp/temp/scaled_img.bmp");	

		for(int k=0;k < 360; k+=step_rot_Z) {
			QImage rotated_img = im_rotate_c(scaled_img, double(k) );

			long double sum = estimate_integral_under_contour(  rotated_img, qv_contour_array[tid], templates_param_hash.value(tid).w);

			float norm_sum = normalize_integral( sum, qv_contour_array[tid].size(), (float)0.7);

			energy_map.insertMulti( tid, qMakePair( k,  norm_sum) );
			// append info about image to auxiliary vector: id			
		}
	}
	
	return 0;
}

int fly_object_classifier::estimate_similarity_rx_ry(QImage const & img) {
	QImage img_grad = gradXY_Sobel( img );

	// fit in frame!!!

	// fit_in_closest_frame( img_grad );

	// get_nearest_templates
	QVector<int> work_templates = get_nearest_templates(templates_param_hash, img_grad.size(), template_similarity);

	printf("get_nearest_templates - ok!, work_templates.size() = %d\n",work_templates.size() );

	// calc contour integral and (or) integral under object
	for(int i_template = 0; i_template < work_templates.size(); ++i_template) {
		if(img_grad.width() == 0) {
			printf("Division by zero - empty gradient image!\n");
			return 0;
		}	
		float scale = float(templates_param_hash.value( work_templates[i_template]).w) / float (img_grad.width() );
		// printf("scale = %f\n", scale);
		// int tid = templates_param_hash.value(work_templates[i]).id;
		int tid = work_templates[i_template];

		QImage scaled_img = im_scale_c(img_grad, scale);
		// scaled_img.save("scaled_img.bmp");
		// printf("scaled_img type = %d\n", scaled_img.format() );
		
		int cur_w,cur_h;
		for(int k=0;k < 360; k += step_rot_Z) {
			QImage rotated_img = im_rotate_c_2(scaled_img, double(k) );
			// rotated_img.save(QString("temp/rotated_img_%1_%2.bmp").arg(i_template).arg(k));
			// fit_in_closest_frame( rotated_img);
			// rotated_img.save("rotated_img_fit.bmp");
			
			for( int ii=(-roi_h); ii<=roi_h; ii+=step_r) {
				for( int jj=(-roi_w); jj<=roi_w; jj+=step_r) {
					cur_h = rotated_img.height();
					cur_w = rotated_img.width();
					int x = jj;
					int y = ii;
					if(ii < 0) y = 0;
					if(jj< 0) x = 0;
					if(ii> 0) cur_h -= ii;
					if(jj > 0) cur_w -= jj;
					QImage rotated_img_cut = rotated_img.copy(jj,ii,cur_w,cur_h);
					// rotated_img_cut.save(QString("temp/rotated_img_cut_%1_%2_%3_%4.bmp").arg(i_template).arg(k).arg(ii).arg(jj));
					// rotated_img_cut.save("rotated_img_cut.bmp");

					long double sum = estimate_integral_under_contour(  rotated_img_cut, qv_contour_array[tid], templates_param_hash.value(tid).w);

					float norm_sum = normalize_integral( sum, qv_contour_array[tid].size(), (float)0.7);

					energy_map.insertMulti( tid, qMakePair( k,  norm_sum) );
					// append info about image to auxiliary vector: id			
				}
			}
		}
	}
	
	return 0;
}

// Store n best elements only
int fly_object_classifier::estimate_similarity_rx_ry_n_elements(QImage const & img) {
	QVector<float> temp_energies; // store n best energies

	QImage img_grad = gradXY_Sobel( img );

	// fit in frame
	// fit_in_closest_frame( img_grad );

	// get_nearest_templates
	QVector<int> work_templates = get_nearest_templates(templates_param_hash, img_grad.size(), template_similarity);

	printf("get_nearest_templates - ok!, work_templates.size() = %d\n",work_templates.size() );

	// calc contour integral and (or) integral under object
	for(int i_template = 0; i_template < work_templates.size(); ++i_template) {
		if(img_grad.width() == 0) {
			printf("Division by zero - empty gradient image!\n");
			return 0;
		}	
		float scale = float(templates_param_hash.value( work_templates[i_template]).w) / float (img_grad.width() );
		int tid = work_templates[i_template];

		QImage scaled_img = im_scale_c(img_grad, scale);
		
		for(int k=0;k < 360; k += step_rot_Z) {
			QImage rotated_img = im_rotate_c_2(scaled_img, double(k) );
			truncate(rotated_img, 25);
			// printf("rotated_img: x1 = %d, y1 = %d, x2 = %d, y2 = %d\n", rotated_img.rect().left(), rotated_img.rect().top(), rotated_img.rect().right(), rotated_img.rect().bottom());
			QRect rect = find_non_zero_rect(rotated_img);
			// printf("rect_coords: x1 = %d, y1 = %d, x2 = %d, y2 = %d\n", rect.left(), rect.top(), rect.right(), rect.bottom());
			rotated_img = rotated_img.copy(rect);
			// rotated_img.save(QString("D:/projects/TOUCH/fly_run/temp/input_img_rot/rotated_img_cut_%1.bmp").arg(k));
			// printf("rotated_img cut: x1 = %d, y1 = %d, x2 = %d, y2 = %d\n", rotated_img.rect().left(), rotated_img.rect().top(), rotated_img.rect().right(), rotated_img.rect().bottom());

			long double w_sum = weighted_full_sum(rotated_img);

			for( int ii=(-roi_h); ii<=roi_h; ii+=step_r) {
				for( int jj=(-roi_w); jj<=roi_w; jj+=step_r) {
					int cur_h = rotated_img.height();
					int cur_w = rotated_img.width();
					int x = jj;
					int y = ii;
					if(ii < 0) y = 0;
					if(jj< 0) x = 0;
					if(ii> 0) cur_h -= ii;
					if(jj > 0) cur_w -= jj;
					QImage rotated_img_cut = rotated_img.copy(jj,ii,cur_w,cur_h);

					long double sum = estimate_integral_under_contour(  rotated_img_cut, qv_contour_array[tid], templates_param_hash.value(tid).w);

					// float norm_sum = normalize_integral( sum, qv_contour_array[tid].size(), 0.7);
					float norm_sum = normalize_integral( sum, w_sum, 1);
					printf("norm_sum = %f, w_sum = %lf, sum = %lf \n", norm_sum, w_sum, sum);

					float old_best_energy;
					if(temp_energies.isEmpty())
						old_best_energy = 0;
					else
						old_best_energy = temp_energies.last();

					if( norm_sum > old_best_energy ) {
						float bad_energy = append_and_pop( temp_energies, this->n_best_elements, norm_sum);
						printf("best = %f, bad_energy = %f\n", norm_sum, bad_energy);
						energy_map.insertMulti( tid, qMakePair( k,  float(norm_sum) ) );
						if(temp_energies.size() >= this->n_best_elements-1) {
							// remove element with too small energy value
							QList<int>	keys = energy_map.keys( qMakePair( k,  bad_energy));
							for(int i=0;i<keys.size();++i) {
								QHash<int, QPair<int, float> >::iterator iter = this->energy_map.find(keys[i]);
								while(iter!=this->energy_map.end() && iter.key()==keys[i]) {
									if(iter.value()==qMakePair( k,  bad_energy)) {
										iter = this->energy_map.erase(iter);
									}
									else {
										++iter;
									}
								}
							}		
						}
					}
					// append info about image to auxiliary vector: id
				}
			}
		}
	}
	
	return 0;
}

void remove_low_energy( QHash< QPair<int,int>,  float> &map, float low_element) {
	// remove element with too small energy value from the energy_map
	QList<QPair<int,int> >	keys = map.keys( low_element);
	for(int i=0;i < keys.size();++i) {
		QHash<QPair<int, int>, float>::iterator iter = map.find(keys[i]);
		while( iter!=map.end() && iter.key()==keys[i]) {
			if( iter.value()== low_element ) {
				iter = map.erase(iter); // remove
			}
			else {
				++iter;
			}
		}
	}		
}

// Store n best elements only
int fly_object_classifier::estimate_similarity_rx_ry_n_elements_eigen_rot(QImage const & img) {
	QVector<float> temp_energies; // store n best energies

	QImage img_grad = gradXY_Sobel( img );

	// find nearest templates by w/h ratio
	img_grad = rotate_to_horizontal(img_grad);
	truncate(img_grad, 15);
	QRect rect = find_non_zero_rect(img_grad);
	img_grad = img_grad.copy(rect);
	QVector<int> work_templates = get_nearest_templates(templates_param_hash, img_grad.size(), template_similarity);
	printf("get_nearest_templates - ok!, work_templates.size() = %d\n",work_templates.size() );

	// calc contour integral and (or) integral under object
	for(int i_template = 0; i_template < work_templates.size(); ++i_template) {
		if(img_grad.width() == 0) {	printf("Division by zero - empty gradient image!\n"); return 0;}	
		float scale = float(templates_param_hash.value( work_templates[i_template]).w) / float (img_grad.width() );
		int tid = work_templates[i_template];
		QImage scaled_img = im_scale_c(img_grad, scale);
		
		for(int k=0;k < 360; k += 180) {// step_rot_Z) {
			double angle_rad = k * double(M_PI) / double(180);
			QImage rotated_img = im_rotate_c_2(scaled_img, angle_rad);
			// rotated_img.save("D:/projects/TOUCH/fly_run/temp/comparision/orig.bmp");
			// QRect rect = find_non_zero_rect(rotated_img);
			// rotated_img = rotated_img.copy(rect);
			long double w_sum = weighted_full_sum(rotated_img);

			for( int ii=(-roi_h); ii <= roi_h; ii+=step_r) {
				for( int jj=(-roi_w); jj <= roi_w; jj+=step_r) {
					// int cur_h = rotated_img.height();
					// int cur_w = rotated_img.width();
					// int x = jj;
					// int y = ii;
					// if(ii < 0) y = 0;
					// if(jj < 0) x = 0;
					// if(ii > 0) cur_h -= ii;
					// if(jj > 0) cur_w -= jj;
					// QImage rotated_img_cut = rotated_img.copy(jj,ii,cur_w,cur_h);

					// testing print
					// rotated_img.save("D:/projects/TOUCH/fly_run/temp/comparision/orig.bmp");
					// QImage test_decompress = decompress_sparse_object( qv_contour_array[tid], templates_param_hash.value(tid).w, templates_param_hash.value(tid).h, 255, 255 );
					// test_decompress.save("D:/projects/TOUCH/fly_run/temp/comparision/template.bmp");
					// getchar();

					long double sum = estimate_integral_under_contour(  rotated_img, qv_contour_array[tid], templates_param_hash.value(tid).w, jj, ii);

					// float norm_sum = normalize_integral( sum, qv_contour_array[tid].size(), 0.7);
					float norm_sum = normalize_integral( sum, w_sum, 1);

					float old_best_energy;
					if(temp_energies.isEmpty())
						old_best_energy = 0;
					else
						old_best_energy = temp_energies.last();

					if( norm_sum > old_best_energy ) {
						float low_energy = append_and_pop( temp_energies, this->n_best_elements, norm_sum);
						energy_hash_table.insert( qMakePair( tid, k),  float(norm_sum) );
						if(temp_energies.size() >= this->n_best_elements-1) {
							remove_low_energy( this->energy_hash_table, low_energy);
						}
					}
					// append info about image to auxiliary vector: id
				}
			}
		}
	}
	return 0;
}

// Store n best elements only
int fly_object_classifier::estimate_similarity_rx_ry_n_elements_eigen_rot_Nicodim_dist(QImage const & img) {
	QVector<float> temp_energies; // store n best energies
	
	// Rotate according eigenvalues
	QImage img_eig = rotate_to_horizontal(img);
	QRect rect = find_non_zero_rect(img_eig); img_eig = img_eig.copy(rect);
	
	// find nearest templates by w/h ratio
	QVector<int> work_templates = get_nearest_templates(templates_param_hash, img_eig.size(), template_similarity);
	printf("get_nearest_templates - ok!, work_templates.size() = %d\n", work_templates.size() );

	// calc contour integral and (or) integral under object
	for(int i_template = 0; i_template < work_templates.size(); ++i_template) {
		if(img_eig.width() == 0) {	printf("Division by zero - empty gradient image!\n"); return 0;}	
		
		float scale = float(templates_param_hash.value( work_templates[i_template]).w) / float (img_eig.width() );
		QImage scaled_img = im_scale_c(img_eig, scale);
		
		int tid = work_templates[i_template];
		
		for(int k=0;k < 360; k += 180) {// step_rot_Z) {
			QImage rotated_img = im_rotate_c_2( scaled_img, k * double(M_PI) / double(180) );
			
			for( int ii=(-roi_h); ii <= roi_h; ii+=step_r) {
				for( int jj=(-roi_w); jj <= roi_w; jj+=step_r) {

					long double sum = nikodim_distance( rotated_img, qv_object_array[tid], templates_param_hash.value(tid).w, jj, ii);
					
					float old_best_energy = temp_energies.isEmpty() ? FLT_MAX : temp_energies.last();
					if( sum < old_best_energy ) { // search min distance
						float low_energy = append_and_pop( temp_energies, this->n_best_elements, sum);
						energy_hash_table.insert( qMakePair( tid, k),  float(sum) );
						if(temp_energies.size() >= this->n_best_elements-1) {
							remove_low_energy( this->energy_hash_table, low_energy);
						}
					}
					// append info about image to auxiliary vector: id
				}
			}
		}
	}
	return 0;
}

// Store n best elements only
int fly_object_classifier::estimate_similarity_rx_ry_n_elements_eigen_rot_Nicodim_dist_temp(QImage const & img) {
	QVector<float> temp_energies; // store n best energies
	// img.save("D:/projects/TOUCH/fly_run/temp/comparision/orig_input.bmp");
	// find nearest templates by w/h ratio
	QImage img_eig = rotate_to_horizontal(img);
	// img_eig.save("D:/projects/TOUCH/fly_run/temp/comparision/img_eig.bmp");
	QRect rect = find_non_zero_rect(img_eig);
	img_eig = img_eig.copy(rect);
	// img_eig.save("D:/projects/TOUCH/fly_run/temp/comparision/img_eig_non_zero.bmp");
	QVector<int> work_templates = get_nearest_templates(templates_param_hash, img_eig.size(), template_similarity);
	printf("get_nearest_templates - ok!, work_templates.size() = %d\n",work_templates.size() );

	// calc contour integral and (or) integral under object
	for(int i_template = 0; i_template < work_templates.size(); ++i_template) {
		if(img_eig.width() == 0) {	printf("Division by zero - empty gradient image!\n"); return 0;}	
		float scale = float(templates_param_hash.value( work_templates[i_template]).w) / float (img_eig.width() );
		int tid = work_templates[i_template];
		QImage scaled_img = im_scale_c(img_eig, scale);
		// scaled_img.save("D:/projects/TOUCH/fly_run/temp/comparision/img_eig_non_zero_scaled.bmp");
		for(int k=0;k < 360; k += 180) {// step_rot_Z) {
			double angle_rad = k * double(M_PI) / double(180);
			QImage rotated_img = im_rotate_c_2(scaled_img, angle_rad);
			// rotated_img.save("D:/projects/TOUCH/fly_run/temp/comparision/img_eig_non_zero_scaled_rotated.bmp");
			// QRect rect = find_non_zero_rect(rotated_img);
			// rotated_img = rotated_img.copy(rect);
			// long double w_sum = weighted_full_sum(rotated_img);
			for( int ii=(-roi_h); ii <= roi_h; ii+=step_r) {
				for( int jj=(-roi_w); jj <= roi_w; jj+=step_r) {
					// testing print
					// rotated_img.save("D:/projects/TOUCH/fly_run/temp/comparision/orig.bmp");
					// QImage test_decompress = decompress_sparse_object( qv_object_array[tid], templates_param_hash.value(tid).w, templates_param_hash.value(tid).h, 255, 255 );
					// test_decompress.save("D:/projects/TOUCH/fly_run/temp/comparision/template.bmp");
					long double sum = nikodim_distance(  rotated_img, qv_object_array[tid], templates_param_hash.value(tid).w, jj, ii);
					// float norm_sum = normalize_integral( sum, qv_contour_array[tid].size(), 0.7);
					float norm_sum = sum;

					float old_best_energy=FLT_MAX;
					if(temp_energies.isEmpty())
						old_best_energy = FLT_MAX;
					else
						old_best_energy = temp_energies.last();

					if( norm_sum < old_best_energy ) { // search min distance
						float low_energy = append_and_pop( temp_energies, this->n_best_elements, norm_sum);
						energy_hash_table.insert( qMakePair( tid, k),  float(norm_sum) );
						if(temp_energies.size() >= this->n_best_elements-1) {
							remove_low_energy( this->energy_hash_table, low_energy);
						}
					}
					// append info about image to auxiliary vector: id
				}
			}
		}
	}
	return 0;
}

cu_fly_object_classifier::cu_fly_object_classifier() {
	set_default_params();

	cu_img_array = 0; // input image
	// cu_img_array_upside_down = 0; // input image rotated by 180 degrees ??
	cu_params_array = 0;
	cu_object_array = 0;
	cu_indices_of_fittest = 0; // is being calculated on CPU and sent to GPU with input image array
	cu_energy_table = 0; // size is 
	cu_energy_table_n_best = 0; // size is n_best_elements
	is_memory_allocated = false;
	is_data_copyed = false;
}

cu_fly_object_classifier::~cu_fly_object_classifier() {
	clearMemory();
}

void cu_fly_object_classifier::clearMemory( ) {
	if(cu_img_array)
		cudaFree(cu_img_array);
	// if(cu_img_array_upside_down)
			// cudaFree(cu_img_array_upside_down);
	if(cu_params_array)
		cudaFree(cu_params_array);
	if(cu_object_array)
			cudaFree(cu_object_array);
	if(cu_indices_of_fittest)
		cudaFree(cu_indices_of_fittest);
	if(cu_energy_table)
			cudaFree(cu_energy_table);
	if(cu_energy_table_n_best)
			cudaFree(cu_energy_table_n_best);
}

int cu_fly_object_classifier::allocate_memory_on_GPU(bool realloc_force) { // allocates required memory on GPU and sets is_memory_allocated to true
	if(is_memory_allocated && realloc_force) {
		printf("realloc_force\n");
		this->clearMemory();
		is_memory_allocated = false;
		this->allocate_memory_on_GPU();
	} 
	else if(is_memory_allocated) {
		printf("Memory was already allocated on GPU\n");
		return ErrorAllocateMemoryGPU;
	}
	else {
		cudaError_t mem2d;
		mem2d = cudaMalloc( (void**)&cu_params_array, sizeof(template_params)*templates_param_hash.size());
		if( mem2d != cudaSuccess ) {
			printf("Error:cudaMalloc((void**)&cu_params_array, sizeof(template_params)*templates_param_hash.size()):Error message=%s\n", cudaGetErrorString(mem2d));
			clearMemory( ); return ErrorAllocateMemoryGPU;
		}
		printf("cu_params_array - allocated, size = %d, sizeof(template_params) = %d\n", templates_param_hash.size(), sizeof(template_params) );
		if(UseIntensity) {
			// check
			if(templates_param_hash.size () != qv_object_array.size () ) {
				printf("cu_fly_object_classifier::allocate_memory_on_GPU: templates_param_hash.size()!=qv_object_array.size()\n");
				clearMemory( ); return ErrorAllocateMemoryGPU;
			}
			template_params t_temp = templates_param_hash.value (qv_object_array.size()-1);
			int size_of_obj_array = t_temp.object_shift + t_temp.compressed_obj_length;
			mem2d = cudaMalloc( (void**) &cu_object_array, sizeof (MAPTYPE)*size_of_obj_array);
			if( mem2d != cudaSuccess ) {
				printf("Error:cudaMalloc((void**)&cu_object_array, sizeof(MAPTYPE)*qv_object_array.size()):Error message=%s\n", cudaGetErrorString(mem2d));
				clearMemory( ); return ErrorAllocateMemoryGPU;
			}
			printf("cu_object_array - allocated, size = %d\n", size_of_obj_array);
			mem2d = cudaMalloc( (void**)&cu_energy_table_n_best, sizeof(float)*n_best_elements);
			if( mem2d != cudaSuccess ) {
				printf("Error:cudaMalloc((void**)&cu_energy_table_n_best, sizeof(float)*n_best_elements):Error message=%s\n", cudaGetErrorString(mem2d));
				clearMemory( ); return ErrorAllocateMemoryGPU;
			}
			printf("cu_energy_table_n_best - allocated, size = %d\n", n_best_elements);
		}
	}
	this->is_memory_allocated = true;
	printf("is_memory_allocated = true\n");
	return 0;
}

int cu_fly_object_classifier::copy_data_to_GPU(bool recopy_force) {
	if (is_data_copyed && recopy_force) {
		printf("recopy_force\n");
		this->clearMemory();
		is_memory_allocated = false;
		this->allocate_memory_on_GPU();
		this->copy_data_to_GPU();
	} else if (is_data_copyed) {
		printf("Data was already copied on GPU\n");
		return ErrorCopyMemoryGPU;
	} else {
		cudaError_t mem2d;
		template_params *temp_params = 0;
		temp_params = listToArray (templates_param_hash.values() );
		if(!temp_params) {
			printf("temp_params == 0\n");
			return -1;
		}
		mem2d = cudaMemcpy( cu_params_array, temp_params, 
			sizeof(template_params)*templates_param_hash.size(), cudaMemcpyHostToDevice);
		if( mem2d != cudaSuccess ) {
			printf("Error:cudaMemcpy(cu_params_array, listToArray(templates_param_hash.values()),sizeof(template_params)*templates_param_hash.size(), cudaMemcpyHostToDevice): Error message=%s\n", 
				cudaGetErrorString(mem2d));
			clearMemory( );	return ErrorCopyMemoryGPU;      
		} ;
		printf("cu_params_array - copyed, size = %d\n", templates_param_hash.size () );
		delete[] temp_params;

		if (UseIntensity) {
			MAPTYPE *obj_array = 0;
			int obj_array_size;
			// int obj_array_size = vecvecToArray(obj_array, qv_object_array);
			obj_array = vecvecToArray (qv_object_array, &obj_array_size);
			if(obj_array==0) {
				printf("obj_array==0\n");
				return -1;
			}
			printf("obj_array_size = %d\n", obj_array_size );
			mem2d = cudaMemcpy( cu_object_array, obj_array, 
				sizeof (MAPTYPE) * obj_array_size, cudaMemcpyHostToDevice);
			if( mem2d != cudaSuccess ) {
				printf("Error:cudaMemcpy (cu_object_array, obj_array, sizeof(MAPTYPE)*obj_array_size, cudaMemcpyHostToDevice): Error message=%s\n", 
					cudaGetErrorString (mem2d) );
				clearMemory ( );	return ErrorCopyMemoryGPU;      
			} ;
			printf("cu_object_array - copyed, size = %d\n", obj_array_size );
			delete[] obj_array;
		}
	}
	
	this->is_data_copyed = true;
	printf("is_data_copyed = true\n");
	return 0;
}

// Store n best elements only
/*int estimate_similarity_rx_ry_n_elements_eigen_rot_Nicodim_dist(QImage const & img) {
	QVector<float> temp_energies; // store n best energies
	
	// find nearest templates by w/h ratio
	QImage img_eig = rotate_to_horizontal(img);
	QRect rect = find_non_zero_rect(img_eig); img_eig = img_eig.copy(rect);

	QVector<int> work_templates = get_nearest_templates(templates_param_hash, img_eig.size(), template_similarity);
	printf("get_nearest_templates - ok!, work_templates.size() = %d\n", work_templates.size() );

	// calc contour integral and (or) integral under object
	for(int i_template = 0; i_template < work_templates.size(); ++i_template) {
		if(img_eig.width() == 0) {	printf("Division by zero - empty gradient image!\n"); return 0;}	
		
		float scale = float(templates_param_hash.value( work_templates[i_template]).w) / float (img_eig.width() );
		QImage scaled_img = im_scale_c(img_eig, scale);
		
		int tid = work_templates[i_template];
		
		for(int k=0;k < 360; k += 180) {// step_rot_Z) {
			QImage rotated_img = im_rotate_c_2( scaled_img, k * double(M_PI) / double(180) );
			
			for( int ii=(-roi_h); ii <= roi_h; ii+=step_r) {
				for( int jj=(-roi_w); jj <= roi_w; jj+=step_r) {

					long double sum = nikodim_distance( rotated_img, qv_object_array[tid], templates_param_hash.value(tid).w, jj, ii);
					
					float old_best_energy = temp_energies.isEmpty() ? FLT_MAX : temp_energies.last();
					if( sum < old_best_energy ) { // search min distance
						float low_energy = append_and_pop( temp_energies, this->n_best_elements, sum);
						energy_hash_table.insert( qMakePair( tid, k),  float(sum) );
						if(temp_energies.size() >= this->n_best_elements-1) {
							remove_low_energy( this->energy_hash_table, low_energy);
						}
					}
					// append info about image to auxiliary vector: id
				}
			}
		}
	}
	return 0;
}*/

int cu_fly_object_classifier::estimate_similarity(QImage const& img) { // recognition
	// Cover GPU memory operation to support CPU only interface
	if( ! this->is_memory_allocated) {this->allocate_memory_on_GPU();} 
	if( ! this->is_data_copyed) {this->copy_data_to_GPU();}
	
	QVector<float> temp_energies(n_best_elements); // store n best energies
	// Rotate input image according eigenvalues
	QImage img_eig = rotate_to_horizontal(img);
	QRect rect = find_non_zero_rect(img_eig); img_eig = img_eig.copy(rect);
	// Find nearest templates by w/h ratio
	QVector<int> work_templates = get_nearest_templates (templates_param_hash, img_eig.size(), template_similarity);
	printf("get_nearest_templates - ok!, work_templates.size() = %d\n", work_templates.size() );

	cudaError_t mem2d;
	// Allocate and copy memory on GPU for indices of fittest and for returning energy values
	mem2d = cudaMalloc( (void**)&cu_indices_of_fittest, sizeof(int)*(work_templates.size() ));
	if( mem2d != cudaSuccess ) {
		printf("Error:cudaMalloc( (void**)&cu_indices_of_fittest, sizeof(int)*work_templates.size():Error message=%s\n", cudaGetErrorString(mem2d));
		clearMemory( ); return -1;
	}
	printf("cu_indices_of_fittest - allocated, size = %d\n", work_templates.size() );

	mem2d = cudaMalloc( (void**)&cu_energy_table, sizeof(float)*(work_templates.size()*(2*roi_h/step_r)*(2*roi_w/step_r) ));
	if( mem2d != cudaSuccess ) {
		printf("Error:cudaMalloc( (void**)&cu_energy_table, sizeof(float)*(work_templates.size()*(2*roi_h/step_r)*(2*roi_w/step_r):Error message=%s\n", cudaGetErrorString(mem2d));
		clearMemory( ); return -1;
	}
	printf("cu_energy_table - allocated, size = %d\n", work_templates.size()*(2*roi_h/step_r)*(2*roi_w/step_r) );
	
	mem2d = cudaMalloc( (void**)&cu_img_array, sizeof(IMGTYPE)*(img_eig.width() * img_eig.height() ) );
	if( mem2d != cudaSuccess ) {
		printf("Error:cudaMalloc( (void**)&cu_img_array, sizeof(IMGTYPE)*(img_eig.width() * img_eig.height() ):Error message=%s\n", cudaGetErrorString(mem2d));
		clearMemory( ); return -1;
	}
	printf("cu_img_array - allocated, size = %d\n", (img_eig.width() * img_eig.height() ) );

	mem2d = cudaMemcpy( cu_indices_of_fittest, work_templates.data(), sizeof(int) * work_templates.size(), cudaMemcpyHostToDevice);
	if( mem2d != cudaSuccess ) {
		printf("Error:cudaMemcpy( cu_indices_of_fittest, work_templates.data(), sizeof(int) * work_templates.size(), cudaMemcpyHostToDevice): Error message=%s\n", cudaGetErrorString(mem2d) );
		clearMemory( ); return -1;      
	} ;
	printf("cu_indices_of_fittest - copyed, size = %d\n", work_templates.size() );
	/*if(work_templates.data()!=0)
		if( cu_base::copy_dataToGPU( cu_indices_of_fittest, (int*)work_templates.data(), sizeof(int) * work_templates.size() ) ) { clearMemory( ); return ErrorCopyMemoryGPU;}
	else
		printf("work_templates.data()==0\n");*/
	IMGTYPE * img_array = 0;
	img_array = new IMGTYPE[img_eig.width() * img_eig.height()];
	if( convert_qimage_to_array( img_array, img_eig) ) return -1;
	if(!img_array) {
		printf("img_array == 0;\n");
		clearMemory( ); return -1;
	}
	mem2d = cudaMemcpy( cu_img_array, img_array, sizeof(IMGTYPE) * img_eig.width() * img_eig.height(), cudaMemcpyHostToDevice);
	if( mem2d != cudaSuccess ) {
		printf("Error:cudaMemcpy( cu_img_array, img_array, sizeof(IMGTYPE) * img_eig.width() * img_eig.height(), cudaMemcpyHostToDevice): Error message=%s\n", cudaGetErrorString(mem2d) );
		clearMemory( ); return -1;      
	} ;
	printf("cu_img_array - copyed, size = %d\n", img_eig.width() * img_eig.height() );
	delete[] img_array;

	unsigned int img_obj_area = areaBW(img_eig);
	
	int NUM_BLOCK_X = 1;
	int NUM_BLOCK_Y = work_templates.size();
    int NUM_THREAD_X = (2*roi_h/step_r)*(2*roi_w/step_r);
	int NUM_THREAD_Y = 1;

	QTime RunTimer;  // Timer
	RunTimer.start ( ) ;

	estimate_nicodim_metric_wrapper (NUM_BLOCK_X, NUM_BLOCK_Y, NUM_THREAD_X, NUM_THREAD_Y,
		cu_img_array, img_eig.width (), img_eig.height (), img_obj_area, cu_object_array, cu_params_array, cu_indices_of_fittest, roi_w, roi_h, step_r, cu_energy_table, cu_energy_table_n_best, n_best_elements);

	fprintf (stdout, "Time of run func call = %lf seconds\n", double (RunTimer.elapsed( ) ) / 1000. ) ;

	energy_table.resize (n_best_elements);
	mem2d = cudaMemcpy (energy_table.data (), cu_energy_table_n_best, sizeof (float) * n_best_elements, cudaMemcpyDeviceToHost);
	if (mem2d != cudaSuccess) {
		printf ("Error:cudaMemcpy (energy_table.data (), cu_energy_table_n_best, sizeof (float) * n_best_elements, cudaMemcpyDeviceToHost): Error message=%s\n", cudaGetErrorString(mem2d) );
		clearMemory ( ); return -1;      
	} ;
	printf ("energy_table - copyed back to CPU, size = %d\n", n_best_elements );
	printf ("The result is %f\n", energy_table[0] );

	return 0;
}
/*int cu_fly_object_classifier::append_template(QImage img, int model_id, float angleX, float angleY, float angleZ) {
	img = rotate_to_horizontal(img);
	// img.save(QString("D:/projects/TOUCH/fly_run/temp/templates_fit_rot_enhance/t_%1_%2_%3_eig.bmp").arg(angleX).arg(angleY).arg(angleZ) );
	QRect rect = find_non_zero_rect(img);
	// rect.setCoords(rect.x()-2, rect.y()-2, rect.width()+4, rect.height()+4 );
	// enhance_rectangle( rect, 2, img.rect() );
	img = img.copy(rect);
	// img.save(QString("D:/projects/TOUCH/fly_run/temp/templates_fit_rot_enhance/t_%1_%2_%3.bmp").arg(angleX).arg(angleY).arg(angleZ) );
	
	int compressed_obj_length = 0;
	if(UseIntensity) {
		// compress (from sparse to compact representation) // store object only
		// append compressed template
		qv_object_array.append(compress_sparse_image(img));
		compressed_obj_length = qv_object_array.last().size();
	}

	// int tid = templates_param_vector.size()+1; // id of template
	int tid = templates_param_hash.size(); // id of template

	// append to vector of templates
	// templates_param_vector.push_back(template_params(tid, angleX, angleY, angleZ, img.width(), img.height(), contour_length ));
	templates_param_hash.insert(tid, template_params(tid, angleX, angleY, angleZ, img.width(), img.height(), 0, compressed_obj_length ));

	// sort by width/height ratio
	// qSort( templates_param_vector.begin(), templates_param_vector.end(), compare_wh_ratio);
	
	return 0;
} */

/*
void remove_low_energy(QHash<int, QPair<int, float>  > &map, float low_element) {
	// remove element with too small energy value from the energy_map
	QList<int>	keys = map.keys( qMakePair( k,  low_element));
	for(int i=0;i<keys.size();++i) {
		QHash<int, QPair<int, float> >::iterator iter = map.find(keys[i]);
		while(iter!=map.end() && iter.key()==keys[i]) {
			if(iter.value()==qMakePair( k,  low_element)) {
				iter = map.erase(iter); // remove
			}
			else {
				++iter;
			}
		}
	}		
}
*/

/*QHash<QObject *, int> objectHash;
...
QHash<QObject *, int>::iterator i = objectHash.find(obj);
while (i != objectHash.end() && i.key() == obj) {
	if (i.value() == 0) {
		i = objectHash.erase(i);
	} else {
		++i;
	}
}

QList<int>	keys = energy_map.keys( qMakePair( k,  bad_energy));
for(int i=0;i<keys.size();++i) {
	energy_map.remove(keys[i]);
}*/
