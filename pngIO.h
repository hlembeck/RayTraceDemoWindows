#pragma once

#include <vector>
#include "lodepng.h"

bool savePNG(unsigned char* rgbQuadArr, unsigned int width, unsigned int height, char* filename);