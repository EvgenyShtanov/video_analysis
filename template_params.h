#ifndef TEMPLATE_PARAMS_H
#define TEMPLATE_PARAMS_H

#include <stdio.h>
#include <stdlib.h>

struct template_params {
		int id; // Order number
		float angleX; // Roll?
		float angleY; // Pitch?
		float angleZ; // Yaw
		int w;
		unsigned int compressed_obj_length; // Number of elements in template, area of template (in pixels)
		unsigned int object_shift; // Shift from the beginning of templates array
		template_params () {
			
		}
		template_params (int tid, float aX, float aY, float aZ, int tw, int col = 0, int o_shift = 0 ) {
			id = tid;
			angleX = aX;
			angleY = aY;
			angleZ = aZ;
			w = tw;
			compressed_obj_length = col;
			object_shift = o_shift;
		}
};

#endif