#ifndef EIGEN_INDEX_VEC_COL_MAJOR
#define EIGEN_INDEX_VEC_COL_MAJOR

#include <ovv/Cluster.h>

class EigenIndexVecColMajor
{
    public:
        EigenIndexVecColMajor(Eigen::MatrixXd &x):x_(x){};
        bool comp(int j, int k) const {return x_(j) < x_(k);}
        bool gt(int j, int k) const {return x_(j) > x_(k);}
    private:
        Eigen::MatrixXd x_;
};

#endif
