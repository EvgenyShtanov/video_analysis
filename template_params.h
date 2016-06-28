#ifndef TEMPLATE_PARAMS_H
#define TEMPLATE_PARAMS_H


struct template_params {
		int id; // Order number
		float angleX; // Roll?
		float angleY; // Pitch?
		float angleZ; // Yaw
		int w;
		int h;
		unsigned int contour_length;
		unsigned int contour_shift;
		unsigned int compressed_obj_length; // Number of elements in template, area of template (in pixels)
		unsigned int object_shift; // Shift from the beginning of templates array
		unsigned long int bit_mask_arg_up;
		unsigned long int bit_mask_arg_down;
		double *hu_moments;
		template_params () {
			this->contour_length = 0;
		}
		template_params (int tid, float aX, float aY, float aZ, int tw, int th, unsigned long int bm_u, unsigned long int bm_d, int cl = 0, int col = 0, int c_shift = 0, int o_shift = 0 ) {
			id = tid;
			angleX = aX;
			angleY = aY;
			angleZ = aZ;
			w = tw;
			h = th;
			bit_mask_arg_up = bm_u;
			bit_mask_arg_down = bm_d;
			contour_length = cl;
			compressed_obj_length = col;
			contour_shift = c_shift;
			object_shift = o_shift;
		}
		template_params (int tid, float aX, float aY, float aZ, int tw, int th, unsigned long int bm_u, unsigned long int bm_d, double *hu, int cl = 0, int col = 0, int c_shift = 0, int o_shift = 0 ) {
			id = tid;
			angleX = aX;
			angleY = aY;
			angleZ = aZ;
			w = tw;
			h = th;
			bit_mask_arg_up = bm_u;
			bit_mask_arg_down = bm_d;
			contour_length = cl;
			compressed_obj_length = col;
			contour_shift = c_shift;
			object_shift = o_shift;
			hu_moments = hu;
		}
};

/**
	Structure for templates sorting by hamming distance between bitmasks
*/
struct bit_mask_param {
	int id;
	int ham_dist;
	bit_mask_param (int id_in=0, int h_d=0) {
		id = id_in;
		ham_dist = h_d;
	}
};

/**
	Structure for templates sorting by distance between Hu's moments
*/
struct hu_dist_param {
	int id;
	double hu_distance;
	hu_dist_param (int id_in = 0, double hu_dist_in = 0.0) {
		id = id_in;
		hu_distance = hu_dist_in;
	}
};


#endif