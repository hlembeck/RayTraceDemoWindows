#include "RayTrace.cuh"
#include <windows.h>

bool getPinholeBitmap(ImgParamPinhole& params, Scene& scene, unsigned char* &rgbQuadArr);