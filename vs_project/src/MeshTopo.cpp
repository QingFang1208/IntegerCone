#include "MeshTopo.h"
using OpenMesh::Vec3i;

MeshTopo::MeshTopo(const Mesh& mesh) : vN(mesh.n_vertices()), fN(mesh.n_faces()), hN(mesh.n_halfedges()), 
	vb(vN, false), h2f(hN,-1), h2v(hN, -1), f2v(fN, Vec3i(-1)), f2f(fN, Vec3i(-1)), f2h(fN, Vec3i(-1))
{
	for (auto hh : mesh.halfedges())
	{
		h2f[hh.idx()] = mesh.face_handle(hh).idx();
		h2v[hh.idx()] = mesh.to_vertex_handle(hh).idx();
		
		vb[h2v[hh.idx()]] = vb[h2v[hh.idx()]] || (h2f[hh.idx()] < 0);
	}

	for (auto fh : mesh.faces())
	{
		auto fv_it(mesh.cfv_iter(fh));
		f2v[fh.idx()][0] = fv_it->idx();		++fv_it;
		f2v[fh.idx()][1] = fv_it->idx();		++fv_it;
		f2v[fh.idx()][2] = fv_it->idx();

		auto fh_it(mesh.cfh_iter(fh));
		f2h[fh.idx()][0] = fh_it->idx();		++fh_it;
		f2h[fh.idx()][1] = fh_it->idx();		++fh_it;
		f2h[fh.idx()][2] = fh_it->idx();

		f2f[fh.idx()][0] = h2f[f2h[fh.idx()][0] ^ 1];
		f2f[fh.idx()][1] = h2f[f2h[fh.idx()][1] ^ 1];
		f2f[fh.idx()][2] = h2f[f2h[fh.idx()][2] ^ 1];
	}
}

void phfToE(const MeshTopo& topo, std::vector<OpenMesh::Vec3i>& phf2e)
{
	phf2e.clear();
	phf2e.assign(topo.h2f.size(), Vec3i(-1));

	for (auto fh : topo.f2h)
	{
		if (topo.h2f[fh[0] ^ 1] >= 0)
		{
			phf2e[fh[0]][0] = fh[0] >> 1;
			phf2e[fh[0]][1] = fh[2] >> 1;
			phf2e[fh[0] ^ 1][2] = fh[1] >> 1;
		}

		if (topo.h2f[fh[1] ^ 1] >= 0)
		{
			phf2e[fh[1]][0] = fh[1] >> 1;
			phf2e[fh[1]][1] = fh[0] >> 1;
			phf2e[fh[1] ^ 1][2] = fh[2] >> 1;
		}

		if (topo.h2f[fh[2] ^ 1] >= 0)
		{
			phf2e[fh[2]][0] = fh[2] >> 1;
			phf2e[fh[2]][1] = fh[1] >> 1;
			phf2e[fh[2] ^ 1][2] = fh[0] >> 1;
		}
	}
}