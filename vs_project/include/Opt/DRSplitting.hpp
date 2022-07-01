#ifndef DRSPLITTING_HPP
#define DRSPLITTING_HPP

#include <eigen3/Eigen/Eigen>
#include "AndersonAcceleration.h"
#include <iostream>

template<bool n> struct AA_trait {};

template<typename Tensor, bool useAA> class DRSplitting
{
protected:
	virtual Tensor proxSubjectFun(const Tensor&) = 0;
	virtual Tensor proxObjectFun(const Tensor&) = 0;
	virtual double norm(const Tensor&) = 0;
	virtual bool record(const Tensor&,int iter) = 0;
	virtual void updateGamma(int count, double res) = 0;
	virtual bool checksuccess() = 0;
	
	void initAnderson(AA_trait<false>) {}
	void initAnderson(AA_trait<true>) {
		AA = std::make_unique<AndersonAcceleration>(andersM, 2*S.size(), S.size());
		AA->init(S, U);
		res = DBL_MAX;
		res_RC = DBL_MAX;
	}

	void resetAnderson(AA_trait<false>) {}
	void resetAnderson(AA_trait<true>) {
		assert(AA != NULL);
		AA->reset(S, U);
	}
	
	bool resSelection(AA_trait<false>) { S = S + S_D; U = proxSubjectFun(S); return true; }
	bool resSelection(AA_trait<true>) {
		if (res < res_RC || reset_AA == true) {
			S = S + S_D;
			U = proxSubjectFun(S);
			res_RC = res;
			S_RC = S;
			U_RC = U;
			AA->compute(S_RC, U_RC, S, U);
			reset_AA = false;
			return true;
		}
		else {
			S = S_RC;
			U = U_RC;
			reset_AA = true;
			AA->reset(S, U);
			return false;
		}
	}

	void init(double gamma_, double thres_, int maxIters_, int andersM_) {
		gamma = gamma_; thres = thres_; iterNum = maxIters_; andersM = andersM_;
	}

public:
	DRSplitting() {}

	Tensor run(const Tensor& A)
	{
		S = A;
		U = proxSubjectFun(S);
		initAnderson(AA_trait<useAA>());

		int count = 0;
		while (count < iterNum && res > thres)
		{
			V = proxObjectFun(2 * U - S);

			S_D = V - U;
			res = norm(S_D);

			if (!resSelection(AA_trait<useAA>())) res = res_RC;

			//if (count % 500 == 0) std::cout << "Residual : " << res << std::endl;

			if (record(U, count)) break;
			updateGamma(count, res);
			count++;
		}
		std::cout << "\nIter num : " << count << "   Residual : " << res << std::endl;

		return U;
	}
protected:
	double gamma;
	double thres;
	int iterNum;
	int andersM;

private:
	double res = DBL_MAX;
	double res_RC = DBL_MAX;
	bool reset_AA = true;

	Tensor S_D;
	Tensor S_RC, U_RC;

	std::unique_ptr<AndersonAcceleration> AA;
	Tensor U, V, S;
};

#endif // !DRSPLITTING_HPP
