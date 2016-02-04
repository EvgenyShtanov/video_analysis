#include "scene_renderer.h"

/*CSceneRenderer::CSceneRenderer() {

} ;

CSceneRenderer::~CSceneRenderer() {

} ;*/

int CSceneRenderer::create_model( s_SceneParam scp, s_ModelViewParam smvp){
	int is_er=0;
	wgt.init_model( smvp.w_view, smvp.h_view, smvp.view_pix_size) ;
	printf("smvp.w_view = %d, smvp.h_view = %d, smvp.view_pix_size = %f, smvp.is_visible = %d\n", smvp.w_view, smvp.h_view, smvp.view_pix_size, smvp.is_visible);
	wgt.set_visible(smvp.is_visible);
	printf("Model was created: w=%d, h=%d, ps = %f\n", smvp.w_view, smvp.h_view, smvp.view_pix_size);
	wgt.set_light_color_to_def( ) ;

	set_scene_param( scp);

	return is_er;
}

void CSceneRenderer::set_scene_param( s_SceneParam scp) {
	wgt.set_cam( scp.angle_camera_azimuth, scp.angle_camera_height);
	wgt.set_light_pos( scp.angle_light_height, scp.angle_light_azimuth);

	wgt.set_img_pix_size( scp.pixel_size, scp.otstup);
}

void CSceneRenderer::set_model_param( s_ModelParam smp) {
	wgt.open_obj( QString("%1/%2.ASE").arg(smp.QSmodel_path).arg(smp.QSname));
}

void CSceneRenderer::set_rotation(float angleX, float  angleY, float angleZ, bool is_update_view){ //!< Perform rendering. angleZ - top view rotation, yaw angle. angleZ - ракурс, рысканье
	// int is_er=0;

	wgt.set_obj_racurs( angleZ, is_update_view );
	wgt.set_obj_rotationX( angleX, is_update_view ); // rotation round X axis
	wgt.set_obj_rotationY( angleY, is_update_view ); // rotation round Y axis
}
/**
	Performes rendering and return rendered image
*/
QImage CSceneRenderer::get_image(int obj_val, bool use_obj, int shadow_val, bool use_shadow, int back_val) {
	QImage img = wgt.get_color_image(obj_val, use_obj, shadow_val, use_shadow, back_val) ;

	return img;
}

QImage CSceneRenderer::get_image() {
	QImage img = wgt.get_color_image() ;

	return img;
}

QVector3D CSceneRenderer::get_object_size() {
	const obj_size* objsize = NULL;
	objsize = wgt.get_object_size( ) ;

	if( objsize == NULL ) {
		printf( "objsize == NULL\n" );
		return QVector3D();
	}
	// printf("\nobjsize->x %f,objsize->y %f,objsize->z %f,mp.pixel_size %f\n",objsize->x,objsize->y,objsize->z,mp.pixel_size);
	return QVector3D(objsize->x,objsize->y,objsize->z);
}
