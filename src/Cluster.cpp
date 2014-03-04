#include <ovv/Cluster.h>

Cluster::Cluster()
{
}

std::ostream& operator<<(std::ostream& output, Cluster& c)
{
    output << c.getID();
    return output;
}

bool Cluster::doDimensionalityReduction(const EigenMatrix &ProjMat)
{
    if(ProjMat.rows() != mean_.rows())
    {
        std::cout << "ProjMat is " << ProjMat.rows() << "x" << ProjMat.cols() << "\n";
        std::cout << "mean is " << mean_.rows() << "x" << mean_.cols() << "\n";
    }

    mean_ = ProjMat.transpose()*mean_;
    //mean_.conservativeResize(ProjMat.cols());    

    cov_ = ProjMat.transpose()*cov_*ProjMat;
    //cov_.conservativeResize(ProjMat.cols(),ProjMat.cols());
    return true;
}
