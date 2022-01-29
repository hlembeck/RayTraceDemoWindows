#pragma once

#include <windows.h>
#include <vector>
#include <string>
#include "resource.h"
#include "RayTraceEntry.cuh"

struct RTParams {
	ImgParamPinhole params;
	Scene scene;
	//Filled by getPinholeBitmap
	unsigned char* rgbQuadArr;
};