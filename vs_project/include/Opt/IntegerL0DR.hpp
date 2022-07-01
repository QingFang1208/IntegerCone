#ifndef INTEGERPL0DR_HPP
#define INTEGERPL0DR_HPP

#include "DRSplitting.hpp"
#include "Candidate.hpp"
#include "OMPHelper.h"
#include <ctime>
/*
solve the following problem:
min ||A1x + b1||_0 + lambda * 0.5 * ||A3x + b3||_2^2, s.t. 
A1x + b1 in {-2^m1, ... ,2^m1} and A2x + b2 in {-2^m2, ... ,2^m2}, ||x||_p<sigma
*/
template <bool useAA> class IntegerL0DR : public DRSplitting<VectorX, useAA>
{
private:
	VectorX proxSubjectFun(const VectorX& t)
	{
		auto tx = t.segment(0, n);
		Eigen::Map<const MatrixXX> ty1(t.data() + n, n1, m1);
		Eigen::Map<const MatrixXX> tz1(t.data() + n + n1 * m1, n1, m1);
		Eigen::Map<const MatrixXX> tr1(t.data() + n + 2 * n1 * m1, n1, m1);

		VectorX u = t;
		auto ux = u.segment(0, n);
		Eigen::Map<MatrixXX> uy1(u.data() + n, n1, m1);
		Eigen::Map<MatrixXX> uz1(u.data() + n + n1 * m1, n1, m1);
		Eigen::Map<MatrixXX> ur1(u.data() + n + 2 * n1 * m1, n1, m1);

		VectorX e1 = VectorX::Ones(n1);
		VectorX s1 = VectorX::Zero(n1);
		for (int i = 0; i < m1; i++)
		{
			uy1.col(i) = ty1.col(i) + tz1.col(i) + tr1.col(i);
			s1 += w1[i] * uy1.col(i);
		}
		s1 /= 3;
		ux = A2Alpha.solve(3 / alpha1 * (A1T * (s1 - b1 - pow(2, m1 - 1) * e1)) + tx);

		s1 = (A1 * ux + b1 + pow(2, m1 - 1) * e1 - s1) / alpha1;
		for (int i = 0; i < m1; i++)
			uy1.col(i) = w1[i] * s1 + uy1.col(i) / 3;
		uz1 = uy1;
		ur1 = uy1;
		return u;
	}

	VectorX proxObjectFun(const VectorX& s)
	{
		VectorX v = s;
		auto sx = s.segment(0, n);
		Eigen::Map<const MatrixXX> sy1(s.data() + n, n1, m1);
		Eigen::Map<const MatrixXX> sz1(s.data() + n + n1 * m1, n1, m1);
		Eigen::Map<const MatrixXX> sr1(s.data() + n + 2 * n1 * m1, n1, m1);

		auto vx = v.segment(0, n);
		Eigen::Map<MatrixXX> vy1(v.data() + n, n1, m1);
		Eigen::Map<MatrixXX> vz1(v.data() + n + n1 * m1, n1, m1);
		Eigen::Map<MatrixXX> vr1(v.data() + n + 2 * n1 * m1, n1, m1);


		v_gamma = (sigma / fmax(sigma, vx.norm()) * vx - vx).cwiseAbs();
		if (vx.norm() > 1)
		{
			lm_gamma = lm_gamma_f / pow(vx.norm(), 0);
		}
		else
		{
			lm_gamma = lm_gamma_f / pow(vx.norm(), 4);
			//vx = vx / vx.norm();
		}
		
		vx = sigma / fmax(sigma, vx.norm()) * vx;
		VectorX e = VectorX::Ones(n1);
		vy1.col(m1 - 1) -= e;
		for (int i = 0; i < n1; i++)
		{
			if (!isMax)
			{
				//vy1.row(i) = L0UpdateY(this->gamma, vy1.row(i).transpose()).transpose();
			}
			vy1.row(i) = L0UpdateY(lm_gamma, vy1.row(i).transpose()).transpose();
		}
		vy1.col(m1 - 1) += e;

		MatrixXX I1 = MatrixXX::Ones(n1, m1);
		MatrixXX dvz1 = vz1 - 0.5 * I1;
		dvz1 = 0.5 * (sqrt(double(n1 * m1)) * dvz1 / dvz1.norm() + I1) - vz1;
		dvz1 = dvz1.cwiseMax(-lm_eps).cwiseMin(lm_eps);
		vz1 = vz1 + dvz1;

		MatrixXX dr1 = vr1.cwiseMax(0).cwiseMin(1) - vr1;
		MatrixXX ddr1 = dr1.cwiseMax(-lm_eps).cwiseMin(lm_eps);
		vr1 = vr1 + ddr1;
		return v;
	}

	double norm(const VectorX& s) { return s.norm(); }

	bool record(const VectorX& u,int iter)
	{
		VectorX K1, iK1, iK;
		iK1 = K1 = A1 * u.segment(0, n) + b1;
		iK.resize(iK1.size());

		for (int i = 0; i < iK1.size(); i++)
		{
			iK1[i] = round(iK1[i]);
/*			if (iK1[i] > 1 + 1e-3 || iK1[i]<-2-1e-3)
			{
				return false;
			}	*/	
		}

		if (fabs(K1.sum() - iK1.sum()) < 0.5)
		{
			iK.segment(0, iK1.size()) = iK1;
			candidates.process(iK, isMax, iter);
			is_success = candidates.is_succ;
			//return candidates.qualified();
		}

		//return false;
		return is_success;
	}

	void updateGamma(int count, double res)
	{
		if (count % 100 == 0)
		{
			std::cout << res << std::endl;
		}
		if ((1+count)%500==0)
		{
			lm_eps += 2e-2;
		}
		if (count == 5000)
		{
			//if (!candidates.qualified())
			//{
			//	lm_gamma_f *= 0.5;
			//	VectorX zero = VectorX::Zero(n);
			//	VectorX s = initOri(zero);
			//	DRSplitting<VectorX, useAA>::restart(s);
			//}
		}
	}
	bool checksuccess()
	{
		lm_gamma -= 3.5e-2;
		return candidates.qualified();
	}
	VectorX L0UpdateY(double gamma, const VectorX& s)
	{
		
		std::vector<double> d(m1 + 1, 0);
		Eigen::Map<VectorX> y(d.data(), m1);

		y = s.cwiseAbs();
		std::sort(d.begin(), d.end(), std::greater<double>());

		if (y.sum() <= gamma) return VectorX::Zero(m1);

		int k = 0;
		double sumd = 0.0;
		for (int i = 1; i < m1 + 1; i++)
		{
			sumd = 0.0;
			for (int j = 0; j < i; j++)
			{
				sumd += d[j];
			}
			if (sumd - i * d[i] >= gamma)
			{
				k = i;
				break;
			}
		}

		double p = -gamma;
		for (int i = 0; i < k; i++)
		{
			p += d[i];
		}
		p /= k;

		y = s.cwiseMax(-p).cwiseMin(p);
		return y;
	}

public:
	IntegerL0DR(int m1_, const ColMajorSparseMatrix A1_, const VectorX b1_, OptCandidates& candidates_, volatile bool& is_success_, double lm_eps_)
		: m1(m1_), n(A1_.cols()), n1(b1_.size()), A1(A1_), b1(b1_), candidates(candidates_), is_success(is_success_), lm_eps(lm_eps_)
	{
		sigma = 1;
		lambda = 0;
		initialize();
		lp = 2;
	}

	void init(double gamma_, double thres_, int maxIters_, int andersM_) {
		DRSplitting<VectorX, useAA>::init(gamma_, thres_, maxIters_, andersM_);
		lm_gamma_f = gamma_;
		gammaN = 100;
		//lm_gamma = gamma_;
		ColMajorSparseMatrix I(n, n);
		I.setIdentity();
		A2Alpha.compute(3 / alpha1 * A1T * A1 + I);
		if (A2Alpha.info() != Eigen::Success)
		{
			printf("IL0 LDLT factory failed\n");
			exit(EXIT_FAILURE);
		}
	}

	VectorX initOri(const VectorX& x0)
	{
		v_gamma.resize(n);
		VectorX s(n + 3 * n1 * m1);
		s.setZero();
		s.segment(0, n) = x0;

		VectorX y0 = A1 * x0 + b1;
		for (int i = 0; i < n1; i++)
		{
			int yi = fmax(round(y0[i]) + pow(2, m1 - 1), 0);
			for (int j = 0; j < m1; j++)
			{
				s(n + n1 * j + i) = yi & 1;
				yi = yi >> 1;
			}
		}
		s.segment(n + 2 * n1 * m1, n1 * m1) = s.segment(n + n1 * m1, n1 * m1) = s.segment(n, n1 * m1);

		return s;
	}

	void run(const VectorX& x0) {
		
		VectorX s = initOri(x0);
		s = DRSplitting<VectorX, useAA>::run(s);
	}

	VectorX runAll(const VectorX& s)
	{
		return DRSplitting<VectorX, useAA>::run(s);
	}

private:
	void initialize()
	{
		//candidates.clear();

		A1T = A1.transpose();
		alpha1 = 0;
		w1 = VectorX::Zero(m1);
		for (int i = 0; i < m1; i++)
		{
			w1[i] = pow(2, i);
			alpha1 += pow(4, i);
		}
	}

private:
	// pardiso coeff
	std::vector<double> rhs;
	bool is_success;
	int m1, n, n1, lp;
	ColMajorSparseMatrix A1, A1T, par_A;
	VectorX b1, b2, b3, v_gamma;
	double sigma;
	double lm_eps;
	//double lm_eps = 0.15;
	double lm_gamma_f = 8e-2;
	double lm_gamma;
	bool isMax = true;
	VectorX w1, w2;
	double alpha1, alpha2;
	double gamma0 = 5e-3;
	double lambda;
	int gammaN;

	double lres = DBL_MAX;
	//double coef = 0.1;

	Eigen::SimplicialLDLT<ColMajorSparseMatrix> A2Alpha;
	
	OptCandidates& candidates;
};

#endif // !INTEGERPL0DR_HPP
