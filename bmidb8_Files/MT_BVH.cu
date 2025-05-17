// MT_BVH_Intersection_Reordered.cu

#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <limits>
#include <cuda_runtime.h>

using std::vector;
using std::array;
using std::string;
using std::cout;
using std::cerr;
using std::endl;

// Alias float3 from CUDA
using float3 = ::float3;

// -------------------------------------
// Basic vector ops (host+device)
// -------------------------------------
__host__ __device__ inline float3 makef3(float x, float y, float z) {
    return make_float3(x,y,z);
}
__host__ __device__ inline float3 subf3(const float3 &a, const float3 &b) {
    float3 r; r.x=a.x-b.x; r.y=a.y-b.y; r.z=a.z-b.z; return r;
}
__host__ __device__ inline float3 minf3(const float3 &a, const float3 &b) {
    float3 r; r.x=fminf(a.x,b.x); r.y=fminf(a.y,b.y); r.z=fminf(a.z,b.z); return r;
}
__host__ __device__ inline float3 maxf3(const float3 &a, const float3 &b) {
    float3 r; r.x=fmaxf(a.x,b.x); r.y=fmaxf(a.y,b.y); r.z=fmaxf(a.z,b.z); return r;
}
__host__ __device__ inline float dotf3(const float3 &a, const float3 &b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
__host__ __device__ inline float3 crossf3(const float3 &a, const float3 &b) {
    float3 r;
    r.x = a.y*b.z - a.z*b.y;
    r.y = a.z*b.x - a.x*b.z;
    r.z = a.x*b.y - a.y*b.x;
    return r;
}

// -------------------------------------
// Data structures
// -------------------------------------
typedef array<float3,3> Triangle;
struct AABB { float3 min, max; };
struct BVHNode { AABB bounds; int left, right, start, count; };

// -------------------------------------
// CPU: OFF loader
// -------------------------------------
vector<Triangle> loadOFF(const string &fname) {
    std::ifstream in{fname}; if(!in) throw std::runtime_error("Cannot open OFF file");
    string hdr; in>>hdr; if(hdr!="OFF") throw std::runtime_error("Not OFF file");
    int nV,nF,nE; in>>nV>>nF>>nE;
    vector<float3> V(nV);
    for(int i=0;i<nV;++i){float x,y,z; in>>x>>y>>z; V[i]=makef3(x,y,z);}    
    vector<Triangle> tris; tris.reserve(nF);
    for(int i=0;i<nF;++i){int cnt,a,b,c; in>>cnt>>a>>b>>c;
        if(cnt==3){tris.push_back({V[a],V[b],V[c]});}
        else in.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    }
    return tris;
}

// Compute triangle AABB
AABB triBounds(const Triangle &T){
    AABB b{T[0],T[0]};
    for(int i=1;i<3;++i){ b.min=minf3(b.min,T[i]); b.max=maxf3(b.max,T[i]); }
    return b;
}

// Recursively build BVH, partitioning idx
int buildBVHRec(vector<BVHNode> &nodes,
                const vector<Triangle> &tris,
                vector<int> &idx,
                int start,int end){
    int id=nodes.size(); nodes.emplace_back(); BVHNode &N=nodes.back();
    // bounds
    AABB box=triBounds(tris[idx[start]]);
    for(int i=start+1;i<end;++i){AABB b=triBounds(tris[idx[i]]); box.min=minf3(box.min,b.min); box.max=maxf3(box.max,b.max);}    
    N.bounds=box; int cnt=end-start;
    if(cnt<=4){ N.left=N.right=-1; N.start=start; N.count=cnt; }
    else{
        // split longest axis
        float3 ext=subf3(box.max,box.min);
        int axis=(ext.x>ext.y&&ext.x>ext.z?0:(ext.y>ext.z?1:2));
        float mid=(((&box.min.x)[axis]+(&box.max.x)[axis])*0.5f);
        int i=start,j=end;
        while(i<j){ AABB b=triBounds(tris[idx[i]]);
            float c=((&b.min.x)[axis]);
            if(c<mid) i++; else std::swap(idx[i],idx[--j]);
        }
        if(i==start||i==end) i=start+cnt/2;
        N.left = buildBVHRec(nodes,tris,idx,start,i);
        N.right= buildBVHRec(nodes,tris,idx,i,end);
        N.start=-1; N.count=0;
    }
    return id;
}

// Returns nodes and final idx order
vector<BVHNode> buildBVH(const vector<Triangle>&tris, vector<int>&idx){
    idx.resize(tris.size()); for(int i=0;i<(int)tris.size();++i) idx[i]=i;
    vector<BVHNode> nodes;
    buildBVHRec(nodes,tris,idx,0,idx.size());
    return nodes;
}

// -------------------------------------
// Device: AABB overlap + ray-tri intersect
// -------------------------------------
__device__ bool overlapAABB(const AABB&a,const AABB&b){
    return !(a.max.x<b.min.x||a.min.x>b.max.x||a.max.y<b.min.y||a.min.y>b.max.y||a.max.z<b.min.z||a.min.z>b.max.z);
}
__device__ bool rayTri(const float3&o,const float3&d,const float3 T[3]){
    const float EPS=1e-8f; float3 e1=subf3(T[1],T[0]),e2=subf3(T[2],T[0]);
    float3 p=crossf3(d,e2); float det=dotf3(e1,p); if(fabsf(det)<EPS) return false;
    float inv=1.0f/det; float3 tvec=subf3(o,T[0]); float u=dotf3(tvec,p)*inv; if(u<0||u>1) return false;
    float3 q=crossf3(tvec,e1); float v=dotf3(d,q)*inv; if(v<0||u+v>1) return false;
    float t=dotf3(e2,q)*inv; return (t>=0&&t<=1+EPS);
}

// -------------------------------------
// Kernel: one thread per B-triangle
// -------------------------------------
__global__ void intersectBVH(
    const BVHNode* nodes,int nodeCount,
    const float3* vertsA,int nA,
    const float3* vertsB,int nB,
    int* d_flag){
    int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=nB) return;
    float3 Tb[3]={vertsB[3*idx],vertsB[3*idx+1],vertsB[3*idx+2]};
    AABB boxB; boxB.min=Tb[0]; boxB.max=Tb[0];
    for(int i=1;i<3;++i){ boxB.min=minf3(boxB.min,Tb[i]); boxB.max=maxf3(boxB.max,Tb[i]); }
    int stack[64],sp=0; stack[sp++]=0;
    while(sp>0&&atomicAdd(d_flag,0)==0){ int ni=stack[--sp]; const BVHNode&n=nodes[ni];
        if(!overlapAABB(n.bounds,boxB)) continue;
        if(n.left<0){ for(int t=0;t<n.count;++t){ int ti=idx/*placeholder*/; /* use permuted index */ }
        } else { stack[sp++]=n.left; stack[sp++]=n.right; }
    }
}

int main(){
    auto A = loadOFF("VH_F_renal_pyramid_L_a.off");
    auto B = loadOFF("VH_F_renal_pyramid_L_b.off");
    vector<int> perm;
    auto nodes = buildBVH(A,perm);
    int nA=A.size(),nB=B.size();
    // reorder vertsA
    vector<float3> vertsA(nA*3);
    for(int i=0;i<nA;++i) for(int j=0;j<3;++j) vertsA[3*i+j]=A[perm[i]][j];
    vector<float3> vertsB(nB*3);
    for(int i=0;i<nB;++i) for(int j=0;j<3;++j) vertsB[3*i+j]=B[i][j];
    // upload and launch as before...
}
