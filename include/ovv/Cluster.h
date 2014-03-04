#ifndef CLUSTERS_H
#define CLUSTERS_H

#include <Eigen/Core>
#include <ovv/SerializationUtils.h>

typedef double size_type;
typedef Eigen::Matrix<size_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenRowMajor;
typedef Eigen::Matrix<size_type, Eigen::Dynamic, 1> EigenColVector;  // did this for sanity check reasons
typedef Eigen::Matrix<size_type, Eigen::Dynamic, 1> EigenVector;  // did this for sanity check reasons
typedef Eigen::Matrix<size_type, Eigen::Dynamic, Eigen::Dynamic> EigenMatrix;  // did this for sanity check reasons
typedef Eigen::Array<size_type, Eigen::Dynamic, Eigen::Dynamic> EigenArray;  // did this for sanity check reasons

//EigenRowMajor constMat = EigenRowMajor::Zero(2,2);
class Cluster
{
    public:
        Cluster();
        Cluster(EigenColVector &mean, EigenMatrix &cov, int npts):mean_(mean),cov_(cov),npts_(npts){};
        Cluster(EigenColVector &mean, EigenMatrix &cov, int npts, int id):mean_(mean),cov_(cov),npts_(npts),m_nID(id){};
        const EigenColVector &centroid(){return mean_;}; // avoid the copy
        const EigenMatrix &covariance(){return cov_;}; // avoid the copy
        int npts(){return npts_;};
        int getID(){return m_nID;};
        bool setID(int id){m_nID = id; return true;};
        friend class VisualWord;
        friend std::ostream& operator <<(std::ostream& output, Cluster &c);         
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive &ar, const unsigned int version)
        {
            ar & m_nID;
            ar & npts_;
            ar & mean_;
            ar & cov_;
        };
        bool doDimensionalityReduction(const EigenMatrix &ProjMat);
     private:
        EigenColVector mean_;
        EigenMatrix cov_;
        int npts_;
        int m_nID;
};
#endif
