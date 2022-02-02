#include "RayTrace.cuh"
#include <windows.h>

bool getPinholeImage(ImgParamPinhole& params, Scene& scene, unsigned char* &rgbQuadArr);