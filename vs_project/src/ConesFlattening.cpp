#include "ConesFlattening.h"
#include "Opt/AlgOpt.h"
#include <fstream>

namespace ConesFlattening
{
	ColMajorSparseMatrix L, A, P,PP;
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

	void interiorMatP(const Mesh& mesh)
	{
		std::vector<Eigen::Triplet<double>> trips;
		trips.reserve(mesh.n_vertices());

		int rowId = 0;
		for (auto v : mesh.vertices())
		{
			if (mesh.is_boundary(v)) continue;
			trips.emplace_back(rowId, v.idx(), 1);
			rowId++;

			if (rowId == mesh.n_vertices() - 1)
			{
				break;
			}
		}

		PP.resize(rowId, mesh.n_vertices());
		PP.setFromTriplets(trips.begin(), trips.end());
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

	void initCoef(const Mesh& mesh, int lp_, double sigma_)
	{
		printf("Initialization\n");
		lp = lp_;
		sigma = sigma_;

		areaMat(mesh);
		interiorMat(mesh);
		interiorMatP(mesh);
		initYamabeCoef(mesh);
	}

	struct pairCmpLess
	{
		inline bool operator ()(std::pair<int, double>& x, std::pair<int, double>& y)
		{
			return fabs(x.second - M_PI_2 * round(x.second * M_2_PI)) < fabs(y.second - M_PI_2 * round(y.second * M_2_PI));
		}
	};

	void geneCone(VectorX& conesK, VectorX& phi, VectorX& MA, int coneMax, double& factorn, int& itern, double& gamma_)
	{
		ColMajorSparseMatrix LN = P * L * A.cwiseInverse() * P.transpose();
		VectorX KN = P * K;
		VectorX Aphi = VectorX::Zero(LN.cols());
		VectorX iphi(LN.cols());
		MA = A.diagonal();
		MA = MA.cwiseAbs2();

		//VectorX init_K = VectorX::Zero(LN.cols());
		//for (size_t i = 0; i < 20; i++)
		//{
		//	int id = 1 + (int)(LN.cols() * rand() / (RAND_MAX + 1.0));
		//	init_K[id] = M_PI_2;
		//}
		//for (size_t i = 0; i < 10; i++)
		//{
		//	int id = 1 + (int)(LN.cols() * rand() / (RAND_MAX + 1.0));
		//	init_K[id] = -M_PI_2;
		//}
		//for (size_t i = 0; i < 5; i++)
		//{
		//	int id = 1 + (int)(LN.cols() * rand() / (RAND_MAX + 1.0));
		//	init_K[id] = -M_PI;
		//}
		//Eigen::SimplicialLDLT<ColMajorSparseMatrix> solver;
		//solver.compute(PP*L*PP.transpose());
		////Aphi = solver.solve(P*init_K - P * K);
		////solver.compute(P*L*P.transpose());
		//iphi = solver.solve(PP*init_K - PP * K);
		//std::cout << "error ====  " << (PP*L*PP.transpose()*iphi - PP * init_K + PP * K).norm() << std::endl;
		//iphi = PP.transpose()*iphi;
		//iphi = iphi - (MA.cwiseProduct(iphi)).sum() / MA.sum() * VectorX::Ones(iphi.size());
		//std::cout << "init bound ==  " << sqrt(MA.transpose() * iphi.cwiseAbs2()) << std::endl;
		// ----- different initial ------
		//iphi = VectorX::Random(LN.cols());
		//iphi = VectorX::Ones(LN.cols());
		//iphi = iphi * 1e6 / iphi.norm();
		//Aphi = A * iphi;
		//std::ofstream uFile("init-u.txt");
		//if (uFile.fail())
		//{
		//	std::cout << "Open " << "init-u.txt" << "failed\n";
		//	exit(EXIT_FAILURE);
		//}

		//for (int i = 0; i < Aphi.size(); ++i)
		//{
		//	uFile << iphi[i] << std::endl;
		//}
		//uFile.close();

		//double eps = 1e-3;
		//std::ofstream conesFile("ran_cone-cones.txt");
		//if (conesFile.fail())
		//{
		//	std::cout << "Open " << "ran_cone-cones.txt" << "failed\n";
		//	exit(EXIT_FAILURE);
		//}

		//for (int i = 0; i < init_K.size(); ++i)
		//{
		//	if (init_K[i] > -eps && init_K[i] < eps) continue;
		//	conesFile << i + 1 << " " << init_K[i] << std::endl;
		//}
		//conesFile.close();

		// ----- different initial ------






		int para_num = 10;
		double gamma = 1e-2;
		double lm_eps = 0.12;
		volatile bool is_success = false;
		std::vector<int> cn(para_num, coneMax);
		std::vector<double> factor(para_num);
		std::vector<VectorX> s(para_num);
		for (size_t i = 0; i < para_num; i++)
		{
			factor[i] = 1.0 - 0.04*i;
		}

		std::vector<OptCandidates*> candidate(para_num, NULL);
		std::vector<IntegerL0DR<false>*> IL(para_num, NULL);

		for (int i = 0; i < para_num; i++)
		{
			candidate[i] = new OptCandidates(P * L * P.transpose() * M_2_PI, MA, KN * M_2_PI, coneMax, sigma);
			//IL[i] = new IntegerL0DR<false>(2, LN * M_2_PI * factor[i] * sigma, KN * M_2_PI, *candidate[i], is_success);
			//s[i] = IL[i]->initOri(Aphi);
			//IL[i]->init(1e-2, 5e-7, 2e4, 0);
			//candidate[i]->factor = factor[i];
		}
		printf("Collect integer candidates\n");
		omp_set_num_threads(para_num);
		for (size_t iter = 0; iter < 8; iter++)
		{
			for (int i = 0; i < para_num; i++)
			{
				IL[i] = new IntegerL0DR<false>(2, LN * M_2_PI * factor[i] * sigma, KN * M_2_PI, *candidate[i], is_success, lm_eps);
				s[i] = IL[i]->initOri(Aphi);
				IL[i]->init(gamma, 5e-7, 2e4, 0);
			}
#pragma omp parallel for
			for (int i = 0; i < para_num; i++)
			{
				s[i] = IL[i]->runAll(s[i]);
			}
			int is_candidate_void = true;
			for (int i = 0; i < para_num; i++)
			{
				delete IL[i];
				if (candidate[i]->is_candidate_void == false)
				{
					iter = 10;
					break;
				}
			}
			gamma *=0.1;
			lm_eps *= 1.5;
		}

		double umin, cmin, bmin = DBL_MAX;
		umin = DBL_MAX;
		cmin = DBL_MAX;
		
		int opt_i;
		for (size_t i = 0; i < para_num; i++)
		{
			if (candidate[i]->qualified() && candidate[i]->E_min<cmin)
			{
				opt_i = i;
				cmin = candidate[i]->E_min;
				bmin = candidate[i]->BB_min;
			}
			if (candidate[i]->qualified() && candidate[i]->E_min == cmin && candidate[i]->BB_min < bmin)
			{
				opt_i = i;
				cmin = candidate[i]->E_min;
				bmin = candidate[i]->BB_min;
			}
		}
		if (!(cmin < DBL_MAX))
		{
			for (int i = 0; i < para_num; i++)
			{
				if (candidate[i]->B_min < umin)
				{
					opt_i = i;
					umin = candidate[i]->B_min;
				}
			}
		}
		phi = P.transpose() * candidate[opt_i]->optOne();
		//conesK = L * phi + P.transpose() * K;
		conesK = P.transpose() *(P*L*P.transpose() * candidate[opt_i]->optOne() + KN);
		factorn = factor[opt_i];
		itern = candidate[opt_i]->opt_iter;
		gamma_ = gamma;
	}
}