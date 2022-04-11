#pragma once
#include "RayTrace.cuh"

struct SceneParams {
	std::vector<double> spectrums;
	std::vector<double> spectrumsBack;
	//Vector of faces instead of meshes to make copying to device faster. In this context, the spd member of face is the index of the spectrums array above to which the face references. Thus, after memcpy to device, a quick kernel call can replace each face's index with the appropriate pointer.
	std::vector<Face> faces;
	//Contains indices of faces that seperate meshes. For instance, if faces contains two triangle meshes F1 and F2, then meshes is the array [0,1] (F1 starts at faceVector[0], and F2 starts at faceVector[1]).
	std::vector<unsigned int> meshes;
};

struct RTParams {
	ImgParamPinhole params;
	SceneParams& sceneParams;
	//Filled by getPinholeBitmap
	unsigned char* rgbQuadArr;
};

bool getPinholeImage(RTParams& rtParams);