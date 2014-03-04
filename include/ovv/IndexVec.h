#ifndef INDEX_VEC
#define INDEX_VEC

#include <ovv/Cluster.h>

class IndexVec 
{
    public:
        IndexVec(const std::vector<size_type> &x):x_(x){};
        bool comp(int j, int k) const {return x_[j] < x_[k];}
    private:
        std::vector<size_type> x_;
};

#endif
