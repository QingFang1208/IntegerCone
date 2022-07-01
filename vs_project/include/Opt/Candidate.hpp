#pragma once

#include "Types.h"
#include <set>

struct cmpLess
{
	inline bool operator()(const VectorX& x, const VectorX& y) const
	{
		int xAbsSum = x.cwiseAbs().cwiseSign().sum();
		int yAbsSum = y.cwiseAbs().cwiseSign().sum();
		if (xAbsSum < yAbsSum) return true;
		else if (xAbsSum == yAbsSum)
		{
			VectorX d = x - y;
			int id = 0;
			while (id < d.size() && d[id] == 0) ++id;
			if (id < d.size() && d[id] < 0) return true;
		}
		return false;
	}
};

struct OptCandidates
{
	std::set<VectorX, cmpLess> iKs;
	double opt_factor, factor;
	int opt_iter;
	ColMajorSparseMatrix S, A;
	VectorX M;
	bool shift = false;
	bool is_succ = false;
	bool is_candidate_void = true;
	int coneMax;
	VectorX b;
	double sigma;
	int lp;
	Eigen::SimplicialLDLT<ColMajorSparseMatrix> SAST;
	VectorX phi_alter, phi_min;
	double B_min = DBL_MAX, E_min = DBL_MAX, BB_min = DBL_MAX;

	OptCandidates(const ColMajorSparseMatrix A_, const VectorX M_, const VectorX b_, int cone_, double sigma_, int lp_ = 2)
		: A(A_), b(b_), M(M_), sigma(sigma_), lp(lp_), coneMax(cone_)
	{
		shift = (A * VectorX::Ones(A.cols())).norm() < 1e-10;
		if (!shift)
		{
			S.resize(A.rows(), A.cols());
			S.setIdentity();
		}
		else
		{
			std::vector<Eigen::Triplet<double>> trips;
			trips.reserve(A.rows());

			int rowId = 0;
			for (int i = 0; i < A.rows() - 1; ++i)
			{
				trips.emplace_back(i, i, 1);
			}

			S.resize(A.rows() - 1, A.cols());
			S.setFromTriplets(trips.begin(), trips.end());
		}

		SAST.compute(S * A * S.transpose());
		if (SAST.info() != Eigen::Success)
		{
			printf("OptCandidate LDLT factory failed\n");
			exit(EXIT_FAILURE);
		}

		clear();

	}	

	void process(const VectorX& iK, bool& isMax, int& iter)
	{
		if (iKs.find(iK) != iKs.end()) return;

		std::cout << "* ";
		iKs.insert(iK);

		VectorX phi = S.transpose() * SAST.solve(S * (iK - b));
		std::cout << sqrt(M.transpose() * phi.cwiseAbs2()) << '\t';

		if (shift)
		{
			if (lp == -1) phi = phi - (phi.maxCoeff() + phi.minCoeff()) / 2 * VectorX::Ones(phi.size());
			else if (lp == 2)
			{
				phi = phi - (M.cwiseProduct(phi)).sum() / M.sum() * VectorX::Ones(phi.size());
			}
		}
		double phiE = iK.cwiseAbs().cwiseSign().sum();
		std::cout << M.cwiseSqrt().cwiseProduct(phi).norm() << "  "<< phiE << std::endl;
		double phiB = lp < 0 ? phi.lpNorm<-1>() : M.cwiseSqrt().cwiseProduct(phi).norm();
		isMax = !(E_min < DBL_MAX);
		if (true)
		{
			is_candidate_void = false;
			if (phiB < B_min)
			{
				//std::cout << "* " << M.cwiseSqrt().cwiseProduct(phi).norm()<< "  " << phiE << std::endl;
				B_min = phiB;
				phi_alter = phi;
				opt_factor = factor;
				opt_iter = iter;
			}

			if (phiB < (sigma + 1e-8) && phiE < E_min)
			{
				//std::cout << "* " << M.cwiseSqrt().cwiseProduct(phi).norm() << "  " << phiE << std::endl;
				E_min = phiE;
				BB_min = phiB;
				phi_min = phi;
				opt_factor = factor;
				opt_iter = iter;
			}
			if (phiB < (sigma + 1e-8) && phiE == E_min && phiB < BB_min)
			{
				//std::cout << "* " << M.cwiseSqrt().cwiseProduct(phi).norm() << "  " << phiE << std::endl;
				E_min = phiE;
				BB_min = phiB;
				phi_min = phi;
				opt_factor = factor;
				opt_iter = iter;
			}
		}
	}

	void clear()
	{
		B_min = DBL_MAX;
		E_min = DBL_MAX;
		BB_min = DBL_MAX;
		iKs.clear();
		opt_factor = 1;
		phi_alter = phi_min = VectorX::Zero(S.cols());
	}
	
	bool qualified() { return E_min < DBL_MAX; }
	bool fail() { return (B_min > sigma || E_min > coneMax); }
	VectorX optOne() { return qualified() ? phi_min : phi_alter; }
};