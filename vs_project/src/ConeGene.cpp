#include <ConeGene.h>
#include "Opt/AlgOpt.h"
ConeGene::ConeGene()
{
}

ConeGene::~ConeGene()
{
}

ColMajorSparseMatrix L, A, P;
VectorX K;

int lp = -1;
double sigma;

void areaMat(const Mesh& mesh)
{
	A.resize(mesh.n_vertices(), mesh.n_vertices());
	if (lp < 0)
	{
		A.setIdentity();
		return;
	}

	std::vector<double> fArea(mesh.n_faces());
	double sumArea = 0;

	for (auto f : mesh.faces())
	{
		OpenMesh::Vec3d p[3];
		auto cfv_it = mesh.cfv_begin(f);
		p[0] = mesh.point(*cfv_it);		++cfv_it;
		p[1] = mesh.point(*cfv_it);		++cfv_it;
		p[2] = mesh.point(*cfv_it);

		fArea[f.idx()] = ((p[0] - p[1]) % (p[1] - p[2])).norm();
		sumArea += fArea[f.idx()];
	}

	std::vector<Eigen::Triplet<double>> trips;
	trips.reserve(mesh.n_vertices());

	for (auto v : mesh.vertices())
	{
		double vArea = 0;
		for (auto vf : mesh.vf_range(v))
		{
			vArea += fArea[vf.idx()];
		}

		vArea /= (3 * sumArea);

		trips.emplace_back(v.idx(), v.idx(), pow(vArea, 1.0 / lp));
	}

	A.setFromTriplets(trips.begin(), trips.end());
}

void interiorMat(const Mesh& mesh)
{
	std::vector<Eigen::Triplet<double>> trips;
	trips.reserve(mesh.n_vertices());

	int rowId = 0;
	for (auto v : mesh.vertices())
	{
		if (mesh.is_boundary(v)) continue;
		trips.emplace_back(rowId, v.idx(), 1);
		rowId++;
	}

	P.resize(rowId, mesh.n_vertices());
	P.setFromTriplets(trips.begin(), trips.end());
}

void initYamabeCoef(const Mesh& mesh)
{
	std::vector<OpenMesh::Vec3i> fv_idx(mesh.n_faces());
	std::vector<OpenMesh::Vec3d> fv_theta(mesh.n_faces()), fv_cot(mesh.n_faces());

	for (auto f : mesh.faces())
	{
		Mesh::VertexHandle v[3];
		auto cfv_it = mesh.cfv_begin(f);
		v[0] = *cfv_it;		++cfv_it;
		v[1] = *cfv_it;		++cfv_it;
		v[2] = *cfv_it;

		OpenMesh::Vec3d e[3];
		e[0] = (mesh.point(v[2]) - mesh.point(v[1])).normalized();
		e[1] = (mesh.point(v[0]) - mesh.point(v[2])).normalized();
		e[2] = (mesh.point(v[1]) - mesh.point(v[0])).normalized();

		fv_idx[f.idx()][0] = v[0].idx();
		fv_idx[f.idx()][1] = v[1].idx();
		fv_idx[f.idx()][2] = v[2].idx();

		fv_theta[f.idx()][0] = acos(-e[1] | e[2]);
		fv_theta[f.idx()][1] = acos(-e[2] | e[0]);
		fv_theta[f.idx()][2] = acos(-e[0] | e[1]);

		fv_cot[f.idx()][0] = -0.5 * (e[1] | e[2]) / (e[1] % e[2]).norm();
		fv_cot[f.idx()][1] = -0.5 * (e[2] | e[0]) / (e[2] % e[0]).norm();
		fv_cot[f.idx()][2] = -0.5 * (e[0] | e[1]) / (e[0] % e[1]).norm();
	}

	K.resize(mesh.n_vertices());
	for (auto v : mesh.vertices())
	{
		if (mesh.is_boundary(v))
			K[v.idx()] = M_PI;
		else
			K[v.idx()] = 2 * M_PI;
	}

	for (auto f : mesh.faces())
	{
		K[fv_idx[f.idx()][0]] -= fv_theta[f.idx()][0];
		K[fv_idx[f.idx()][1]] -= fv_theta[f.idx()][1];
		K[fv_idx[f.idx()][2]] -= fv_theta[f.idx()][2];
	}

	std::vector<Eigen::Triplet<double>> trips;
	trips.reserve(9 * mesh.n_faces());

	for (auto f : mesh.faces())
	{
		trips.emplace_back(fv_idx[f.idx()][0], fv_idx[f.idx()][0], fv_cot[f.idx()][1] + fv_cot[f.idx()][2]);
		trips.emplace_back(fv_idx[f.idx()][1], fv_idx[f.idx()][1], fv_cot[f.idx()][0] + fv_cot[f.idx()][2]);
		trips.emplace_back(fv_idx[f.idx()][2], fv_idx[f.idx()][2], fv_cot[f.idx()][0] + fv_cot[f.idx()][1]);
		trips.emplace_back(fv_idx[f.idx()][0], fv_idx[f.idx()][1], -fv_cot[f.idx()][2]);
		trips.emplace_back(fv_idx[f.idx()][1], fv_idx[f.idx()][0], -fv_cot[f.idx()][2]);
		trips.emplace_back(fv_idx[f.idx()][1], fv_idx[f.idx()][2], -fv_cot[f.idx()][0]);
		trips.emplace_back(fv_idx[f.idx()][2], fv_idx[f.idx()][1], -fv_cot[f.idx()][0]);
		trips.emplace_back(fv_idx[f.idx()][2], fv_idx[f.idx()][0], -fv_cot[f.idx()][1]);
		trips.emplace_back(fv_idx[f.idx()][0], fv_idx[f.idx()][2], -fv_cot[f.idx()][1]);
	}

	L.resize(mesh.n_vertices(), mesh.n_vertices());
	L.setFromTriplets(trips.begin(), trips.end());
}

void ConeGene::initCoef(const Mesh& mesh, int lp_, double sigma_)
{
	printf("Initialization\n");
	lp = lp_;
	sigma = sigma_;

	areaMat(mesh);
	interiorMat(mesh);
	initYamabeCoef(mesh);
}

struct pairCmpLess
{
	inline bool operator ()(std::pair<int, double>& x, std::pair<int, double>& y)
	{
		return fabs(x.second - M_PI_2 * round(x.second * M_2_PI)) < fabs(y.second - M_PI_2 * round(y.second * M_2_PI));
	}
};

void ConeGene::geneCone(VectorX& conesK, VectorX& phi, VectorX& MA, int coneMax, double& factor, int& iter)
{
	//ColMajorSparseMatrix LN = P * L * A.cwiseInverse() * P.transpose();
	//VectorX KN = P * K;
	//VectorX Aphi = VectorX::Zero(LN.cols());
	//MA = A.diagonal();
	//MA = MA.cwiseAbs2();
	//printf("Collect integer candidates\n");

	////double alpha[5] = { 0.075, 7.5e-2, 5e-2, 2.5e-2, 1e-2 };
	//int iterNum[5] = { 2e4, 7e3, 9e3, 1.1e4, 1.3e4 };
	//double alpha = 5e-3;
	//bool is_fail = true;
	//OptCandidates candidateKs(P * L * P.transpose() * M_2_PI, MA, KN * M_2_PI, coneMax, sigma);
	//int iiter = 0;
	//while (iiter < 1 && is_fail)
	//{

	//	IntegerL0DR<false> IL(2, LN * M_2_PI * factor * sigma, KN * M_2_PI, 1, lp, candidateKs);
	//	VectorX s = IL.initOri(Aphi);
	//	candidateKs.factor = factor;
	//	IL.init(alpha, 5e-6, 2e4, 0);
	//	IL.init_mkl_solver();
	//	s = IL.runAll(s);

	//	is_fail = candidateKs.fail();
	//	iiter++;
	//	factor -= 0.1;
	//	alpha *= 0.3;
	//}
	//phi = P.transpose() * candidateKs.optOne();
	//conesK = L * phi + P.transpose() * K;
	//factor = candidateKs.opt_factor;
	//iter = candidateKs.opt_iter;
}