#ifndef EIGEN_INDEX_VEC
#define EIGEN_INDEX_VEC

#include <ovv/Cluster.h>

class EigenIndexVec
{
    public:
        EigenIndexVec(EigenRowMajor &x):x_(x){};
        bool comp(int j, int k) const {return x_(j) < x_(k);}
    private:
        EigenRowMajor x_;
};

#endif
