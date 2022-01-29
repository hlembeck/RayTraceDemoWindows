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
		{-1.0, 1.0, 0.0},
		{1.0, 1.0, 0.0},
		{0.0, 0.0, -1.0},
		spectrum
	};

	mesh->faces[1] = {
		{1.0, -1.0, 0.0},
		{-1.0, -1.0, 0.0},
		{-1.0, 1.0, 0.0},
		{0.0, 0.0, -1.0},
		spectrum
	};
}