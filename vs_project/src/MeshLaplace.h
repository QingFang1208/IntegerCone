#pragma once
#include "MeshTopo.h"
#include <eigen3/Eigen/Eigen>

void faceArea(const OpenMesh::Vec3d* points, const MeshTopo& topo, std::vector<double>& fA);
void faceArea(const Mesh& mesh, std::vector<double>& fA);
void vertArea(const OpenMesh::Vec3d* points, const MeshTopo& topo, std::vector<double>& vA);
void vertArea(const Mesh& mesh, std::vector<double>& vA);
void vertGauss(const OpenMesh::Vec3d* points, const MeshTopo& topo, std::vector<double>& vK);
void vertGauss(const Mesh& mesh, std::vector<double>& vK);
void faceAngle(const OpenMesh::Vec3d* points, const MeshTopo& topo, std::vector<OpenMesh::Vec3d>& f2a);
void faceAngle(const Mesh& mesh, std::vector<OpenMesh::Vec3d>& f2a);
void laplaceTriplet(const OpenMesh::Vec3d* points, const MeshTopo& topo, std::vector<Eigen::Triplet<double>>& trips);
void laplaceTriplet(const Mesh& mesh, std::vector<Eigen::Triplet<double>>& trips);

