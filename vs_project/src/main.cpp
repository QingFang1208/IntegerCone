#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include "ConesFlattening.h"

void printHelp(const std::string& programName)
{
	std::cout << "Help: "
		<< programName
		<< " input_obj_file "
		<< "output_file_basename "
		<< "[--normBound=sigma] "
		<< std::endl;
}

bool doesArgExist(const std::string& arg, const std::string& searchStr)
{
	return arg.find(searchStr) != std::string::npos;
}

bool parseArg(const std::string& arg, const std::string& searchStr, std::string& value)
{
	if (doesArgExist(arg, searchStr)) {
		value = arg.substr(arg.find_first_of(searchStr[searchStr.size() - 1]) + 1);
		return true;
	}
	return false;
}

void parseArgs(int argc, const char* argv[], std::string& objPath, std::string& outPath, double &sigma)
{
	if (argc < 3) {
		// input and/or output path not specified
		printHelp(argv[0]);
		exit(EXIT_FAILURE);

	}
	else {
		// parse arguments
		objPath = argv[1];
		outPath = argv[2];

		std::string tmp;
		for (int i = 2; i < argc; i++) {
			if (parseArg(argv[i], "--normBound=", tmp)) sigma = std::stod(tmp);
		}
	}
}

void saveCones(const VectorX& conesK, std::string conesPath, double eps = 1e-3)
{
	std::ofstream conesFile(conesPath);
	if (conesFile.fail())
	{
		std::cout << "Open " << conesPath << "failed\n";
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < conesK.size(); ++i)
	{
		if (conesK[i] > -eps && conesK[i] < eps) continue;
		conesFile << i + 1 << " " << conesK[i] << std::endl;
	}
	conesFile.close();
}

void saveU(const VectorX& u, std::string uPath)
{
	std::ofstream uFile(uPath);
	if (uFile.fail())
	{
		std::cout << "Open " << uPath << "failed\n";
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < u.size(); ++i)
	{
		uFile << u[i] << std::endl;
	}
	uFile.close();
}

void saveInfo(double factor, double t, double l2norm, int iter, double gamma, std::string infoPath)
{
	std::ofstream infoFile(infoPath);
	if (infoFile.fail())
	{
		std::cout << "Open " << infoPath << "failed\n";
		exit(EXIT_FAILURE);
	}

	infoFile << "u L2 norm : \t" << l2norm << std::endl;
	infoFile << "factor : \t" << factor << std::endl;
	infoFile << "iter : \t" << iter << std::endl;
	infoFile << "gamma : \t" << gamma << std::endl;
	infoFile << "cost time : \t" << t << " s\n";
	infoFile.close();
}

int main(int argc, const char *argv[])
{
	std::string objPath = "";
	std::string outPath = "";
	double sigma = 0.15;
	int cN = 8;
	int iter = 0;
	double factor = 1;
	double gamma;
	bool seamless = true;

	parseArgs(argc, argv, objPath, outPath, sigma);

	Mesh mesh;

	std::cout << "load mesh from " << objPath << std::endl;
	if (!MeshTools::ReadMesh(mesh, objPath))
	{
		std::cout << "load failed!\n";
		exit(EXIT_FAILURE);
	}

	VectorX conesK, u, A;
	
	clock_t start, end;
	start = clock();

	ConesFlattening::initCoef(mesh, 2, sigma);
	ConesFlattening::geneCone(conesK, u, A, cN, factor, iter, gamma);


	VectorX u_l2 = u;
	end = clock();
	
	double costTime = (double)(end - start) / CLOCKS_PER_SEC;
	std::cout << "Cost time : " << costTime << " s\n";

	std::string conesPath = outPath + "-cones.txt";
	std::string uPath = outPath + "-u.txt";
	std::string infoPath = outPath + "-info.txt";

	saveCones(conesK, conesPath);
	saveU(u, uPath);
	saveInfo(factor,costTime, sqrt(A.transpose() * u_l2.cwiseAbs2()), iter, gamma, infoPath);

	return 0;
}