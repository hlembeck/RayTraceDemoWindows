#include "pngIO.h"

template <typename T> void fillVector(T* arr, unsigned int len, std::vector<T> &vec) {
	for (unsigned int i = 0; i < len; i++) {
		vec.push_back(arr[i]);
	}
}

bool savePNG(unsigned char* rgbQuadArr, unsigned int width, unsigned int height, char* filename) {
	std::vector<unsigned char> tmp = {};
	fillVector<unsigned char>(rgbQuadArr, width*height, tmp);
	unsigned error;
	if ((error = lodepng::encode(filename, rgbQuadArr, width, height))) {
		printf("encoder error %s\n", lodepng_error_text(error));
		return false;
	}
	return true;
}