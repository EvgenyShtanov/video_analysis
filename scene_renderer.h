#ifndef __SCENERENDERER_H__
#define __SCENERENDERER_H__

#include <QImage>
#include <QString>
#include <QTime>
#include <QWidget>
#include <QDir>
#include <stdio.h>
#include <math.h>
#include <QVector>
#include <QVector3D>
#include <QSize>

#include <iostream>
#include "touch_export.h"
#include "model.h"
// #include "cuda_properties.h"

struct TOUCH_DLL s_SceneParam {
	float angle_light_height; //!< углы солнца
	float angle_light_azimuth;
	float angle_camera_height; //!< углы расположения камеры, по умолчанию (0,0)
	float angle_camera_azimuth;
	float pixel_size;  //!< размер пикселя. Rendered image pixel size
	float otstup; // ќтступ от проекции модели в выходном изображении. Ќеобходим дл€ отображени€ тени. Additional border in rendered image for shadow.
	s_SceneParam(){
		angle_camera_height = 0; //углы расположения камеры, по умолчанию (0,0)
		angle_camera_azimuth = 0;
		otstup = (float)0.25;
	};
};

struct TOUCH_DLL s_ModelParam {
	QString QSname; // имя модели (.ASE файла)
	QString QSmodel_path; // путь, где лежат .ASE файлы
	QVector3D object_size;
	s_ModelParam() {
		QSname = QString();
		QSmodel_path = QString();
		object_size = QVector3D();
	}
};


struct TOUCH_DLL s_ModelViewParam {
	int w_view;
	int	h_view; // 
	float view_pix_size;
	bool is_visible;
	s_ModelViewParam() {
		w_view=10;
		h_view=10;
		view_pix_size = ( float ) 0.2;    
		is_visible = false;
	};
};

class TOUCH_DLL CSceneRenderer {
	public:;
		CSceneRenderer () {};
		virtual ~CSceneRenderer ()  {;};

		int create_model( s_SceneParam scp, s_ModelViewParam smvp = s_ModelViewParam());
		void set_scene_param( s_SceneParam scp) ;
		void set_model_param( s_ModelParam smp) ;
		
		void set_rotation(float angleX, float  angleY, float angleZ, bool is_update_view = false); //!< Perform rendering. angleZ - top view rotation, yaw angle. angleZ - ракурс, рысканье
		QImage get_image(int obj_val, bool use_obj, int shadow_val, bool use_shadow, int back_val);
		QImage get_image();
		QVector3D get_object_size();
	private:;
		model wgt;
};

#endif
