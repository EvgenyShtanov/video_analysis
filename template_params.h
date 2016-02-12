#ifndef TEMPLATE_PARAMS_H
#define TEMPLATE_PARAMS_H


struct template_params {
		int id; // order number
		float angleX;
		float angleY;
		float angleZ;
		int w;
		int h;
		unsigned int contour_length;
		unsigned int contour_shift;
		unsigned int compressed_obj_length;
		unsigned int object_shift;
		template_params() {
		this->contour_length = 0;
	}
	template_params (int tid, float aX, float aY, float aZ, int tw, int th, int cl = 0, int col = 0, int c_shift = 0, int o_shift = 0 ) {
		id = tid;
		angleX = aX;
		angleY = aY;
		angleZ = aZ;
		w = tw;
		h = th;
		contour_length = cl;
		compressed_obj_length = col;
		contour_shift = c_shift;
		object_shift = o_shift;
	}
};

#endif