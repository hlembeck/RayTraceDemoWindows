#pragma once

#include <string>
#include "resource.h"
#include "RayTraceEntry.cuh"
#include "ParamsMenu.h"
#include "pngIO.h"

struct RTParams {
	ImgParamPinhole params;
	Scene scene;
	//Filled by getPinholeBitmap
	unsigned char* rgbQuadArr;
};