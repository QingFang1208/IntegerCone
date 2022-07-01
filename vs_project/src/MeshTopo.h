#pragma once
#include "MeshDefinition.h"

struct MeshTopo
{
	int vN, fN, hN;
	std::vector<bool> vb;
	std::vector<int> h2f, h2v;
	std::vector<OpenMesh::Vec3i> f2v, f2f, f2h;

	MeshTopo(const Mesh& mesh);
};

void phfToE(const MeshTopo& topo, std::vector<OpenMesh::Vec3i>& phf2e);
