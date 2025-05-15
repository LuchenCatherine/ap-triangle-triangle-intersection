#pragma once
#include <vector>
#include <fstream>
#include <stdexcept>
#include <string>

struct Vec3 {
    float x,y,z;
};

using TriangleVec = std::vector<Vec3>;

inline TriangleVec loadOFF(const std::string& path){
    std::ifstream in(path);
    if(!in) throw std::runtime_error("Cannot open "+path);
    std::string hdr;
    in>>hdr;
    if(hdr!="OFF") throw std::runtime_error(path+" is not OFF");
    size_t nv,nf,ne;
    in>>nv>>nf>>ne;
    std::vector<Vec3> verts(nv);
    for(size_t i=0;i<nv;++i)
        in>>verts[i].x>>verts[i].y>>verts[i].z;
    TriangleVec tris; tris.reserve(nf*3);
    for(size_t f=0; f<nf; ++f){
        int vc; in>>vc;
        std::vector<size_t> idx(vc);
        for(int k=0;k<vc;++k) in>>idx[k];
        for(int k=1;k+1<vc;++k){
            tris.push_back(verts[idx[0]]);
            tris.push_back(verts[idx[k]]);
            tris.push_back(verts[idx[k+1]]);
        }
    }
    return tris;
}



