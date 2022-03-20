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
		spectrum
	};

	mesh->faces[1] = {
		{1.0, -1.0, 0.0},
		{-1.0, 1.0, 0.0},
		{-1.0, -1.0, 0.0},
		{0.0, 0.0, 1.0},
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
		spectrum
	};
	mesh->faces[1] = {
		{1.0,-1.0,-1.0},
		{-1.0,-1.0,-1.0},
		{-1.0,1.0,-1.0},
		{0.0,0.0,-1.0},
		spectrum
	};
	//Front face
	mesh->faces[2] = {
		{1.0,-1.0,1.0},
		{-1.0,1.0,1.0},
		{1.0,1.0,1.0},
		{0.0,0.0,1.0},
		spectrum
	};
	mesh->faces[3] = {
		{1.0,-1.0,1.0},
		{-1.0,-1.0,1.0},
		{-1.0,1.0,1.0},
		{0.0,0.0,1.0},
		spectrum
	};
	//Top face
	mesh->faces[4] = {
		{1.0,1.0,1.0},
		{1.0,1.0,-1.0},
		{-1.0,1.0,-1.0},
		{0.0,1.0,0.0},
		spectrum
	};
	mesh->faces[5] = {
		{1.0,1.0,1.0},
		{-1.0,1.0,-1.0},
		{-1.0,1.0,1.0},
		{0.0,1.0,0.0},
		spectrum
	};
	//Bottom face
	mesh->faces[6] = {
		{1.0,-1.0,1.0},
		{-1.0,-1.0,-1.0},
		{1.0,-1.0,-1.0},
		{0.0,-1.0,0.0},
		spectrum
	};
	mesh->faces[7] = {
		{1.0,-1.0,1.0},
		{-1.0,-1.0,1.0},
		{-1.0,-1.0,-1.0},
		{0.0,-1.0,0.0},
		spectrum
	};
	//Left face
	mesh->faces[8] = {
		{1.0,-1.0,-1.0},
		{1.0,1.0,-1.0},
		{1.0,1.0,1.0},
		{1.0,0.0,0.0},
		spectrum
	};
	mesh->faces[9] = {
		{1.0,-1.0,-1.0},
		{1.0,1.0,1.0},
		{1.0,-1.0,1.0},
		{1.0,0.0,0.0},
		spectrum
	};
	//Right face
	mesh->faces[10] = {
		{-1.0,-1.0,-1.0},
		{-1.0,1.0,1.0},
		{-1.0,1.0,-1.0},
		{-1.0,0.0,0.0},
		spectrum
	};
	mesh->faces[11] = {
		{-1.0,-1.0,-1.0},
		{-1.0,1.0,1.0},
		{-1.0,1.0,-1.0},
		{-1.0,0.0,0.0},
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
		spectrum
	};

	faces[1] = {
		{1.0, -1.0, 0.0},
		{-1.0, -1.0, 0.0},
		{-1.0, 1.0, 0.0},
		{0.0, 0.0, -1.0},
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
		spectrum
	};
	faces[1] = {
		{1.0,-1.0,-1.0},
		{-1.0,-1.0,-1.0},
		{-1.0,1.0,-1.0},
		{0.0,0.0,-1.0},
		spectrum
	};
	//Front face
	faces[2] = {
		{1.0,-1.0,1.0},
		{-1.0,1.0,1.0},
		{1.0,1.0,1.0},
		{0.0,0.0,1.0},
		spectrum
	};
	faces[3] = {
		{1.0,-1.0,1.0},
		{-1.0,-1.0,1.0},
		{-1.0,1.0,1.0},
		{0.0,0.0,1.0},
		spectrum
	};
	//Top face
	faces[4] = {
		{1.0,1.0,1.0},
		{1.0,1.0,-1.0},
		{-1.0,1.0,-1.0},
		{0.0,1.0,0.0},
		spectrum
	};
	faces[5] = {
		{1.0,1.0,1.0},
		{-1.0,1.0,-1.0},
		{-1.0,1.0,1.0},
		{0.0,1.0,0.0},
		spectrum
	};
	//Bottom face
	faces[6] = {
		{1.0,-1.0,1.0},
		{-1.0,-1.0,-1.0},
		{1.0,-1.0,-1.0},
		{0.0,-1.0,0.0},
		spectrum
	};
	faces[7] = {
		{1.0,-1.0,1.0},
		{-1.0,-1.0,1.0},
		{-1.0,-1.0,-1.0},
		{0.0,-1.0,0.0},
		spectrum
	};
	//Left face
	faces[8] = {
		{1.0,-1.0,-1.0},
		{1.0,1.0,-1.0},
		{1.0,1.0,1.0},
		{1.0,0.0,0.0},
		spectrum
	};
	faces[9] = {
		{1.0,-1.0,-1.0},
		{1.0,1.0,1.0},
		{1.0,-1.0,1.0},
		{1.0,0.0,0.0},
		spectrum
	};
	//Right face
	faces[10] = {
		{-1.0,-1.0,-1.0},
		{-1.0,1.0,1.0},
		{-1.0,1.0,-1.0},
		{-1.0,0.0,0.0},
		spectrum
	};
	faces[11] = {
		{-1.0,-1.0,-1.0},
		{-1.0,1.0,1.0},
		{-1.0,1.0,-1.0},
		{-1.0,0.0,0.0},
		spectrum
	};
	return Mesh{faces,12};
}

void addPlateHOST(std::vector<Face>& faceVector, double* transformMatrix, unsigned int spectrum) {
	Face face, *meshStart;

	face = {
		{1.0, -1.0, 0.0},
		{1.0, 1.0, 0.0},
		{-1.0, 1.0, 0.0},
		{0.0, 0.0, 1.0}
	};
	face.spdIndex = spectrum;
	
	faceVector.push_back(face);

	face = {
		{1.0, -1.0, 0.0},
		{-1.0, 1.0, 0.0},
		{-1.0, -1.0, 0.0},
		{0.0, 0.0, 1.0}
	};
	face.spdIndex = spectrum;

	faceVector.push_back(face);

	meshStart = faceVector.data() + faceVector.size() - 2;

	transformMeshHOST(meshStart,2,transformMatrix);
}

void addPrismHOST(std::vector<Face>& faceVector, double* transformMatrix, unsigned int spectrum) {
	Face face, *meshStart;

	//Back face
	face = {
		{1.0,-1.0,-1.0},
		{-1.0,1.0,-1.0},
		{1.0,1.0,-1.0},
		{0.0,0.0,-1.0},
	};
	face.spdIndex = spectrum;
	faceVector.push_back(face);

	face = {
		{1.0,-1.0,-1.0},
		{-1.0,-1.0,-1.0},
		{-1.0,1.0,-1.0},
		{0.0,0.0,-1.0},
	};
	face.spdIndex = spectrum;
	faceVector.push_back(face);

	//Front face
	face = {
		{1.0,-1.0,1.0},
		{1.0,1.0,1.0},
		{-1.0,1.0,1.0},
		{0.0,0.0,1.0},
	};
	face.spdIndex = spectrum;
	faceVector.push_back(face);

	face = {
		{1.0,-1.0,1.0},
		{-1.0,1.0,1.0},
		{-1.0,-1.0,1.0},
		{0.0,0.0,1.0},
	};
	face.spdIndex = spectrum;
	faceVector.push_back(face);

	//Top face
	face = {
		{1.0,1.0,1.0},
		{1.0,1.0,-1.0},
		{-1.0,1.0,-1.0},
		{0.0,1.0,0.0},
	};
	face.spdIndex = spectrum;
	faceVector.push_back(face);

	face = {
		{1.0,1.0,1.0},
		{-1.0,1.0,-1.0},
		{-1.0,1.0,1.0},
		{0.0,1.0,0.0},
	};
	face.spdIndex = spectrum;
	faceVector.push_back(face);

	//Bottom face
	face = {
		{1.0,-1.0,1.0},
		{-1.0,-1.0,-1.0},
		{1.0,-1.0,-1.0},
		{0.0,-1.0,0.0},
	};
	face.spdIndex = spectrum;
	faceVector.push_back(face);

	face = {
		{1.0,-1.0,1.0},
		{-1.0,-1.0,1.0},
		{-1.0,-1.0,-1.0},
		{0.0,-1.0,0.0},
	};
	face.spdIndex = spectrum;
	faceVector.push_back(face);

	//Left face
	face = {
		{1.0,-1.0,-1.0},
		{1.0,1.0,-1.0},
		{1.0,1.0,1.0},
		{1.0,0.0,0.0},
	};
	face.spdIndex = spectrum;
	faceVector.push_back(face);

	face = {
		{1.0,-1.0,-1.0},
		{1.0,1.0,1.0},
		{1.0,-1.0,1.0},
		{1.0,0.0,0.0},
	};
	face.spdIndex = spectrum;
	faceVector.push_back(face);

	//Right face
	face = {
		{-1.0,-1.0,-1.0},
		{-1.0,1.0,1.0},
		{-1.0,1.0,-1.0},
		{-1.0,0.0,0.0},
	};
	face.spdIndex = spectrum;
	faceVector.push_back(face);

	face = {
		{-1.0,-1.0,-1.0},
		{-1.0,-1.0,1.0},
		{-1.0,1.0,1.0},
		{-1.0,0.0,0.0},
	};
	face.spdIndex = spectrum;
	faceVector.push_back(face);


	meshStart = faceVector.data() + faceVector.size() - 12;
	transformMeshHOST((Face*)meshStart, 12, transformMatrix);
}