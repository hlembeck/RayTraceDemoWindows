#include "StandardMesh.cuh"

__global__ void printMeshes(Mesh* meshes, unsigned int len) {
	for (unsigned int i = 0; i < len; i++) {
		printf("Mesh %d: \n", i);
		for (unsigned int j = 0; j < meshes[i].len; j++) {
			printf(" Face %d:\n", j);
			printf(" [%f %f %f] [%f %f %f] [%f %f %f]\n", meshes[i].faces[j].p1.x, meshes[i].faces[j].p1.y, meshes[i].faces[j].p1.z, meshes[i].faces[j].p2.x, meshes[i].faces[j].p2.y, meshes[i].faces[j].p2.z, meshes[i].faces[j].p3.x, meshes[i].faces[j].p3.y, meshes[i].faces[j].p3.z);
			printf(" Normal: [%f %f %f]\n\n", meshes[i].faces[j].n.x, meshes[i].faces[j].n.y, meshes[i].faces[j].n.z);
		}
	}
}

__global__ void standardPlate(Mesh* mesh, double* spectrum) {
	mesh->faces = new Face[2];
	mesh->len = 2;
	mesh->faces[0] = {
		{1.0, -1.0, 0.0},
		{1.0, 1.0, 0.0},
		{-1.0, 1.0, 0.0},
		{0.0, 0.0, 1.0},
		1,
		1,
		spectrum,
	};

	mesh->faces[1] = {
		{1.0, -1.0, 0.0},
		{-1.0, 1.0, 0.0},
		{-1.0, -1.0, 0.0},
		{0.0, 0.0, 1.0},
		1,
		1,
		spectrum
	};
}

__global__ void standardPrism(Mesh* mesh, double* spectrum) {
	mesh->faces = new Face[12];
	mesh->len = 12;
	//Back face
	mesh->faces[0] = {
		{1.0,-1.0,-1.0},
		{-1.0,1.0,-1.0},
		{1.0,1.0,-1.0},
		{0.0,0.0,-1.0},
		1,
		1,
		spectrum
	};
	mesh->faces[1] = {
		{1.0,-1.0,-1.0},
		{-1.0,-1.0,-1.0},
		{-1.0,1.0,-1.0},
		{0.0,0.0,-1.0},
		1,
		1,
		spectrum
	};
	//Front face
	mesh->faces[2] = {
		{1.0,-1.0,1.0},
		{-1.0,1.0,1.0},
		{1.0,1.0,1.0},
		{0.0,0.0,1.0},
		1,
		1,
		spectrum
	};
	mesh->faces[3] = {
		{1.0,-1.0,1.0},
		{-1.0,-1.0,1.0},
		{-1.0,1.0,1.0},
		{0.0,0.0,1.0},
		1,
		1,
		spectrum
	};
	//Top face
	mesh->faces[4] = {
		{1.0,1.0,1.0},
		{1.0,1.0,-1.0},
		{-1.0,1.0,-1.0},
		{0.0,1.0,0.0},
		1,
		1,
		spectrum
	};
	mesh->faces[5] = {
		{1.0,1.0,1.0},
		{-1.0,1.0,-1.0},
		{-1.0,1.0,1.0},
		{0.0,1.0,0.0},
		1,
		1,
		spectrum
	};
	//Bottom face
	mesh->faces[6] = {
		{1.0,-1.0,1.0},
		{-1.0,-1.0,-1.0},
		{1.0,-1.0,-1.0},
		{0.0,-1.0,0.0},
		1,
		1,
		spectrum
	};
	mesh->faces[7] = {
		{1.0,-1.0,1.0},
		{-1.0,-1.0,1.0},
		{-1.0,-1.0,-1.0},
		{0.0,-1.0,0.0},
		1,
		1,
		spectrum
	};
	//Left face
	mesh->faces[8] = {
		{1.0,-1.0,-1.0},
		{1.0,1.0,-1.0},
		{1.0,1.0,1.0},
		{1.0,0.0,0.0},
		1,
		1,
		spectrum
	};
	mesh->faces[9] = {
		{1.0,-1.0,-1.0},
		{1.0,1.0,1.0},
		{1.0,-1.0,1.0},
		{1.0,0.0,0.0},
		1,
		1,
		spectrum
	};
	//Right face
	mesh->faces[10] = {
		{-1.0,-1.0,-1.0},
		{-1.0,1.0,1.0},
		{-1.0,1.0,-1.0},
		{-1.0,0.0,0.0},
		1,
		1,
		spectrum
	};
	mesh->faces[11] = {
		{-1.0,-1.0,-1.0},
		{-1.0,1.0,1.0},
		{-1.0,1.0,-1.0},
		{-1.0,0.0,0.0},
		1,
		1,
		spectrum
	};
}

Mesh standardPlateHOST(double* spectrum) {
	Face* faces = new Face[2];
	faces[0] = {
		{1.0, -1.0, 0.0},
		{-1.0, 1.0, 0.0},
		{1.0, 1.0, 0.0},
		{0.0, 0.0, -1.0},
		1,
		1,
		spectrum
	};

	faces[1] = {
		{1.0, -1.0, 0.0},
		{-1.0, -1.0, 0.0},
		{-1.0, 1.0, 0.0},
		{0.0, 0.0, -1.0},
		1,
		1,
		spectrum
	};
	return Mesh{ faces,2 };
}

Mesh standardPrismHOST(double* spectrum) {
	Face* faces = new Face[12];
	//Back face
	faces[0] = {
		{1.0,-1.0,-1.0},
		{-1.0,1.0,-1.0},
		{1.0,1.0,-1.0},
		{0.0,0.0,-1.0},
		1,
		1,
		spectrum
	};
	faces[1] = {
		{1.0,-1.0,-1.0},
		{-1.0,-1.0,-1.0},
		{-1.0,1.0,-1.0},
		{0.0,0.0,-1.0},
		1,
		1,
		spectrum
	};
	//Front face
	faces[2] = {
		{1.0,-1.0,1.0},
		{-1.0,1.0,1.0},
		{1.0,1.0,1.0},
		{0.0,0.0,1.0},
		1,
		1,
		spectrum
	};
	faces[3] = {
		{1.0,-1.0,1.0},
		{-1.0,-1.0,1.0},
		{-1.0,1.0,1.0},
		{0.0,0.0,1.0},
		1,
		1,
		spectrum
	};
	//Top face
	faces[4] = {
		{1.0,1.0,1.0},
		{1.0,1.0,-1.0},
		{-1.0,1.0,-1.0},
		{0.0,1.0,0.0},
		1,
		1,
		spectrum
	};
	faces[5] = {
		{1.0,1.0,1.0},
		{-1.0,1.0,-1.0},
		{-1.0,1.0,1.0},
		{0.0,1.0,0.0},
		1,
		1,
		spectrum
	};
	//Bottom face
	faces[6] = {
		{1.0,-1.0,1.0},
		{-1.0,-1.0,-1.0},
		{1.0,-1.0,-1.0},
		{0.0,-1.0,0.0},
		1,
		1,
		spectrum
	};
	faces[7] = {
		{1.0,-1.0,1.0},
		{-1.0,-1.0,1.0},
		{-1.0,-1.0,-1.0},
		{0.0,-1.0,0.0},
		1,
		1,
		spectrum
	};
	//Left face
	faces[8] = {
		{1.0,-1.0,-1.0},
		{1.0,1.0,-1.0},
		{1.0,1.0,1.0},
		{1.0,0.0,0.0},
		1,
		1,
		spectrum
	};
	faces[9] = {
		{1.0,-1.0,-1.0},
		{1.0,1.0,1.0},
		{1.0,-1.0,1.0},
		{1.0,0.0,0.0},
		1,
		1,
		spectrum
	};
	//Right face
	faces[10] = {
		{-1.0,-1.0,-1.0},
		{-1.0,1.0,1.0},
		{-1.0,1.0,-1.0},
		{-1.0,0.0,0.0},
		1,
		1,
		spectrum
	};
	faces[11] = {
		{-1.0,-1.0,-1.0},
		{-1.0,1.0,1.0},
		{-1.0,1.0,-1.0},
		{-1.0,0.0,0.0},
		1,
		1,
		spectrum
	};
	return Mesh{faces,12};
}

void addPlateHOST(std::vector<Face>& faceVector, double* transformMatrix, unsigned int spectrum, double reflectivity, double refractiveIndex) {
	Face face, *meshStart;

	face = {
		{1.0, -1.0, 0.0},
		{1.0, 1.0, 0.0},
		{-1.0, 1.0, 0.0},
		{0.0, 0.0, 1.0}
	};
	face.spdIndex = spectrum;
	face.reflectivity = reflectivity;
	face.refractiveIndex = -refractiveIndex;
	
	faceVector.push_back(face);

	face = {
		{1.0, -1.0, 0.0},
		{-1.0, 1.0, 0.0},
		{-1.0, -1.0, 0.0},
		{0.0, 0.0, 1.0}
	};
	face.spdIndex = spectrum;
	face.reflectivity = reflectivity;
	face.refractiveIndex = -refractiveIndex;

	faceVector.push_back(face);

	meshStart = faceVector.data() + faceVector.size() - 2;

	transformMeshHOST(meshStart,2,transformMatrix);
}

void addPrismHOST(std::vector<Face>& faceVector, double* transformMatrix, unsigned int spectrum, double reflectivity, double refractiveIndex) {
	Face face, *meshStart;

	//Back face
	face = {
		{1.0,-1.0,-1.0},
		{-1.0,1.0,-1.0},
		{1.0,1.0,-1.0},
		{0.0,0.0,-1.0},
	};
	face.spdIndex = spectrum;
	face.reflectivity = reflectivity;
	face.refractiveIndex = -refractiveIndex;
	faceVector.push_back(face);

	face = {
		{1.0,-1.0,-1.0},
		{-1.0,-1.0,-1.0},
		{-1.0,1.0,-1.0},
		{0.0,0.0,-1.0},
	};
	face.spdIndex = spectrum;
	face.reflectivity = reflectivity;
	face.refractiveIndex = -refractiveIndex;
	faceVector.push_back(face);

	//Front face
	face = {
		{1.0,-1.0,1.0},
		{1.0,1.0,1.0},
		{-1.0,1.0,1.0},
		{0.0,0.0,1.0},
	};
	face.spdIndex = spectrum;
	face.reflectivity = reflectivity;
	face.refractiveIndex = -refractiveIndex;
	faceVector.push_back(face);

	face = {
		{1.0,-1.0,1.0},
		{-1.0,1.0,1.0},
		{-1.0,-1.0,1.0},
		{0.0,0.0,1.0},
	};
	face.spdIndex = spectrum;
	face.reflectivity = reflectivity;
	face.refractiveIndex = -refractiveIndex;
	faceVector.push_back(face);

	//Top face
	face = {
		{1.0,1.0,1.0},
		{1.0,1.0,-1.0},
		{-1.0,1.0,-1.0},
		{0.0,1.0,0.0},
	};
	face.spdIndex = spectrum;
	face.reflectivity = reflectivity;
	face.refractiveIndex = -refractiveIndex;
	faceVector.push_back(face);

	face = {
		{1.0,1.0,1.0},
		{-1.0,1.0,-1.0},
		{-1.0,1.0,1.0},
		{0.0,1.0,0.0},
	};
	face.spdIndex = spectrum;
	face.reflectivity = reflectivity;
	face.refractiveIndex = -refractiveIndex;
	faceVector.push_back(face);

	//Bottom face
	face = {
		{1.0,-1.0,1.0},
		{-1.0,-1.0,-1.0},
		{1.0,-1.0,-1.0},
		{0.0,-1.0,0.0},
	};
	face.spdIndex = spectrum;
	face.reflectivity = reflectivity;
	face.refractiveIndex = -refractiveIndex;
	faceVector.push_back(face);

	face = {
		{1.0,-1.0,1.0},
		{-1.0,-1.0,1.0},
		{-1.0,-1.0,-1.0},
		{0.0,-1.0,0.0},
	};
	face.spdIndex = spectrum;
	face.reflectivity = reflectivity;
	face.refractiveIndex = -refractiveIndex;
	faceVector.push_back(face);

	//Left face
	face = {
		{1.0,-1.0,-1.0},
		{1.0,1.0,-1.0},
		{1.0,1.0,1.0},
		{1.0,0.0,0.0},
	};
	face.spdIndex = spectrum;
	face.reflectivity = reflectivity;
	face.refractiveIndex = -refractiveIndex;
	faceVector.push_back(face);

	face = {
		{1.0,-1.0,-1.0},
		{1.0,1.0,1.0},
		{1.0,-1.0,1.0},
		{1.0,0.0,0.0},
	};
	face.spdIndex = spectrum;
	face.reflectivity = reflectivity;
	face.refractiveIndex = -refractiveIndex;
	faceVector.push_back(face);

	//Right face
	face = {
		{-1.0,-1.0,-1.0},
		{-1.0,1.0,1.0},
		{-1.0,1.0,-1.0},
		{-1.0,0.0,0.0},
	};
	face.spdIndex = spectrum;
	face.reflectivity = reflectivity;
	face.refractiveIndex = -refractiveIndex;
	faceVector.push_back(face);

	face = {
		{-1.0,-1.0,-1.0},
		{-1.0,-1.0,1.0},
		{-1.0,1.0,1.0},
		{-1.0,0.0,0.0},
	};
	face.spdIndex = spectrum;
	face.reflectivity = reflectivity;
	face.refractiveIndex = -refractiveIndex;
	faceVector.push_back(face);


	meshStart = faceVector.data() + faceVector.size() - 12;
	transformMeshHOST((Face*)meshStart, 12, transformMatrix);
}

inline bool isEqual(Triple<double>& a, Triple<double>& b) {
	return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
}

bool isVolume(Face* faces, unsigned int len) {
	/*
	Argument of correctness

	Regardless of orientability, a mesh that encloses a volume (rather, at least one volume -- a mesh can enclose multiple volumes) must have the property that every edge of each face must be covered by another distinct face. By requiring that there is a pair of faces that do not lie on the same plane, we ensure that some volume is enclosed by the mesh. This function executes a simple algorithm to test this. If the mesh is a volume, the function updates the faces to allow the refractive index to be considered. If not, rays will not split upon hitting the mesh.
	*/

	// (ret & 0x1) is a flag for whether faces[i] has edge [1,2] covered. Similarly for (ret & 0x2) and (ret & 0x4)
	unsigned int ret;
	for (unsigned int i = 0; i < len; i++) {
		ret = 0;
		
		//Check for face adjacent (without regard to orientation) to edge [1,2] of faces[i]
		for (unsigned int j = 0; j < len; j++) {
			if (i == j)
				continue;
			if (isEqual(faces[i].p1, faces[j].p1)) {
				if (isEqual(faces[i].p2, faces[j].p2) || isEqual(faces[i].p2, faces[j].p3)) {
					//Covers face [1,2]
					ret |= 1;
				}
				else if (isEqual(faces[i].p3, faces[j].p2) || isEqual(faces[i].p3, faces[j].p3)) {
					//Covers face [1,3]
					ret |= 4;
				}
				goto next;
			}

			if (isEqual(faces[i].p1, faces[j].p2)) {
				if (isEqual(faces[i].p2, faces[j].p1) || isEqual(faces[i].p2, faces[j].p3)) {
					//Covers face [1,2]
					ret |= 1;
				}
				else if (isEqual(faces[i].p3, faces[j].p1) || isEqual(faces[i].p3, faces[j].p3)) {
					//Covers face [1,3]
					ret |= 4;
				}
				goto next;
			}

			if (isEqual(faces[i].p1, faces[j].p3)) {
				if (isEqual(faces[i].p2, faces[j].p1) || isEqual(faces[i].p2, faces[j].p2)) {
					//Covers face [1,2]
					ret |= 1;
				}
				else if (isEqual(faces[i].p3, faces[j].p1) || isEqual(faces[i].p3, faces[j].p2)) {
					//Covers face [1,3]
					ret |= 4;
				}
				goto next;
			}

			if (isEqual(faces[i].p2, faces[j].p1)) {
				if (isEqual(faces[i].p3, faces[j].p2) || isEqual(faces[i].p3, faces[j].p3)) {
					ret |= 2;
				}
				goto next;
			}

			if (isEqual(faces[i].p2, faces[j].p2)) {
				if (isEqual(faces[i].p3, faces[j].p1) || isEqual(faces[i].p3, faces[j].p3)) {
					ret |= 2;
				}
				goto next;
			}

			if (isEqual(faces[i].p2, faces[j].p3)) {
				if (isEqual(faces[i].p3, faces[j].p1) || isEqual(faces[i].p3, faces[j].p2)) {
					ret |= 2;
				}
				goto next;
			}
		next:
		}

		if (ret != 7) {
			printf("%d %d\n", ret, i);
			return false;
		}
	}
	return true;
}

bool isOrientedVolume(Face* faces, unsigned int len) {
	unsigned int ret;

	for (unsigned int i = 0; i < len; i++) {
		ret = 0;

		for (unsigned int j = 0; j < len; j++) {
			if (i == j)
				continue;
			if (isEqual(faces[i].p1, faces[j].p1)) {
				if (isEqual(faces[i].p2, faces[j].p3)) {
					ret |= 1; 
				}
				else if (isEqual(faces[i].p3, faces[j].p2)) {
					ret |= 4;
				}
			}
			else if (isEqual(faces[i].p1, faces[j].p2)) {
				if (isEqual(faces[i].p2, faces[j].p3)) {
					ret |= 1;
				}
				else if (isEqual(faces[i].p3, faces[j].p3)) {
					ret |= 4;
				}
			}
			else if (isEqual(faces[i].p1, faces[j].p3)) {
				if (isEqual(faces[i].p2, faces[j].p2)) {
					ret |= 1;
				}
				else if (isEqual(faces[i].p3, faces[j].p1)) {
					ret |= 4;
				}
			}
			else if (isEqual(faces[i].p2, faces[j].p2) && isEqual(faces[i].p3,faces[j].p1)) {
				ret |= 2;
			}
			else if (isEqual(faces[i].p2, faces[j].p1) && isEqual(faces[i].p3,faces[j].p3)) {
				ret |= 2;
			}
			else if (isEqual(faces[i].p2, faces[j].p3) && isEqual(faces[i].p3,faces[j].p2)) {
				ret |= 2;
			}

		}

		if (ret != 7) {
			printf("%d %d\n", ret, i);
			return false;
		}
	}


	for (unsigned int i = 0; i < len; i++) {
		faces[i].refractiveIndex *= -1;
	}

	return true;
}