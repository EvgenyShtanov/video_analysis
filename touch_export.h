#ifdef TOUCH_EXPORT_DLL
	#if defined( WIN32 )
		#define TOUCH_DLL     __declspec(dllexport)
	#else
		#define TOUCH_DLL
	#endif
#else// #ifdef TOUCH_EXPORT_DLL
	#if defined( WIN32 )
		#define TOUCH_DLL     __declspec(dllimport)
	#else
		#define TOUCH_DLL
	#endif
#endif// #ifdef TOUCH_EXPORT_DLL

