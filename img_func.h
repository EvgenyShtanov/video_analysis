#ifndef IMG_FUNC_H
#define IMG_FUNC_H

#include <QImage>
#include <QPixmap>
#include <QTime>
#include <QBitmap>

#define IMGTYPE float
// #define IMGTYPE unsigned char

class ImgArray {
private:  
  IMGTYPE*Array;
  int w;
  int h;
public:
  ImgArray(QPixmap map,bool negative=0) {
     // printf("ImgArray constructor\n");
     // QTime Timer2 ;
     // Timer2.start() ;    
     
     w=map.width ();
     h=map.height();
     Array=new IMGTYPE[w*h];

     // fp=fopen("D:/temp/files/touch/out_img.txt","w");
     for(int m=0;m<h;++m) {
        for(int n=0;n<w;++n) {
           if(!negative)
              Array[m*w+n]=qGray(map.toImage().pixel(n,m));
           else
              Array[m*w+n]=255-qGray(map.toImage().pixel(n,m));
              //fprintf(fp,"%d ",imgArray[m*w+n]);
        }
        //fprintf(fp,"\n");
     }    
     // fclose(fp);
     // fprintf(stdout,"The time of the create imgArray = %lf seconds\n", double(Timer2.elapsed()) / 1000. ) ;
  };

  ImgArray(QImage map,bool negative=0) {
     // printf("ImgArray QImage constructor\n");
     // QTime Timer2 ;
     // Timer2.start() ;    
     
     w=map.width ();
     h=map.height();
     Array=new IMGTYPE[w*h];
     //FILE * fp;
      //fp=fopen("D:/temp/files/touch/out_img_array.txt","w");
			     //printf("ImgArray QImage constructor 1\n");
     for(int m=0;m<h;++m) {
        for(int n=0;n<w;++n) {
           if(!negative)
              Array[m*w+n]=qGray(map.pixel(n,m));
           else
              Array[m*w+n]=255-qGray(map.pixel(n,m));
              //printf("%f ", Array[m*w+n]);
              //fprintf(fp,"%f ", Array[m*w+n]);
			   //printf("ImgArray QImage constructor 2 \n");
        }
        //fprintf(fp,"\n");
     }    
      //fclose(fp);
     // fprintf(stdout," time of creation of the imgArray = %lf seconds\n", double(Timer2.elapsed()) / 1000. ) ;
  };
  
  ~ImgArray( ) {
    // printf("ImgArray destructor1\n");
    delete[] Array;
    // printf("ImgArray destructor2\n");
  };  
  int width ( ) { return w ; };
  int height( ) { return h ; };
  IMGTYPE*getArray ( ) { return Array ; };
  QSize getQSize ( ) { return QSize ( w , h ) ; };

  QImage MaptoImage(IMGTYPE*data) {
	  Q_UNUSED( data )
     QImage img(w,h,QImage::Format_Indexed8);
     //for(int m=0;m<h;++m) {
        //for(int n=0;n<w;++n) {
              //Array[m*w+n]=qGray(map.toImage().pixel(n,m));
              //Array[m*w+n]=qGray(map.toImage().pixel(n,m)); 
         //     img.setPixel((unsigned int)Array[m*w+n],n,m);
              //fprintf(fp,"%d ",imgArray[m*w+n]);
        //}
     return img;
  }; 
 // ImgArray(QPixmap map,bool negative=0) {
 // };
};

QImage
convertTo8(QImage const&InImg) {
   QImage curImg ( InImg.size(), QImage::Format_Indexed8 ) ;

   QVector < QRgb > color_table ;
   for( int k = 0 ; k < 256 ; k++ ) color_table << qRgb( k, k, k ) ;
   curImg.setColorTable( color_table ) ;

   if ( InImg.format() != QImage::Format_Indexed8 ) {
      for ( int i = 0; i < InImg.height(); ++i )
         for ( int j = 0; j < InImg.width(); ++j ) {
              //cout << " x, j = " << j << "y, i = " << i << "qGray( InImg.pixel(j,i) ) = " << qGray( InImg.pixel(j,i) ) << endl ;
              curImg.setPixel( j,i, qGray( InImg.pixel(j,i) ) ) ;
           }
      //QVector < QRgb > color_table ;
      //for( int k = 0 ; k < 256 ; k++ ) color_table << qRgb( k, k, k ) ;
      //curImg = curImg.convertToFormat( QImage::Format_Indexed8, color_table ) ;
   }
   else curImg = InImg ;

   return curImg ;
}

template<typename T>
int convert_array_to_qimage( QImage &qimage_out, T const&img_char_grad) { // , int w, int h) {
	// QImage qimage_out( w, h, QImage::Format_Indexed8 );

	// QVector < QRgb > color_table ;
	// for( int k = 0 ; k < 256 ; k++ ) color_table << qRgb( k, k, k ) ;
	// qimage_out.setColorTable( color_table ) ;

	int count =0;
	int w = qimage_out.width();
	int h = qimage_out.height();

	for(int i=0; i < h;++i) {
		for(int j=0; j < w; ++j) {
			// printf("%d ",);
			qimage_out.setPixel( j, i, (int)img_char_grad[count]);
			count++;
		}
	}
	
	return 0;
}

template<typename T>
int convert_qimage_to_array(T*img_char_grad, QImage qimage_in) {

	int w = qimage_in.width();
	int h =  qimage_in.height();

	for ( int i = 0; i < h; ++i )
		for ( int j = 0; j < w; ++j ) {
			//cout << " x, j = " << j << "y, i = " << i << "qGray( InImg.pixel(j,i) ) = " << qGray( InImg.pixel(j,i) ) << endl ;
			 img_char_grad[w*i+j] = (T) qGray( qimage_in.pixel(j,i) );
		}
	
	return 0;
}

#endif
