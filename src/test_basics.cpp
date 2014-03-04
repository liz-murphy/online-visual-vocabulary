#include <iostream>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/lexical_cast.hpp>

#include <matfile/Input>
#include <matfile/Core>

#include <Eigen/Dense>
#include <matfile/Wrappers/EigenDenseIWrapper.hpp>

#include <ovv/Cluster.h>  // for EigenRowMajorjor
#include <ovv/Vocab.h>  // for EigenRowMajorjor

#include <string>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <fstream>

using namespace std;
using namespace matfile;
using namespace boost::numeric::ublas;

int VisualWord::s_nID = 0;

void loadMatlabMatrix(const string &strFilename, const string &strVarName, EigenMatrix &inputMatrix)
{
    InputDevice myFile(strFilename);
    EigenMatrix loadedMatrix = EigenMatrix::Zero(2,2);
    EigenDenseIWrapper<size_type> wrapperDense(loadedMatrix);
    myFile.readArray(strVarName, wrapperDense);
    inputMatrix = loadedMatrix;
}

int main(int argc, char *argv[])
{
    Vocab incremental_vocab;
    incremental_vocab.setNN(3);
    EigenColVector mean = EigenMatrix::Zero(2,1);
    EigenMatrix mean_tmp = EigenMatrix::Zero(2,1);
    EigenMatrix npts_tmp = EigenMatrix::Zero(2,1);
    EigenMatrix cov = EigenMatrix::Zero(2,2);
    EigenColVector npts = EigenMatrix::Zero(2,1);

    incremental_vocab.setDimensionality(2);
    EigenMatrix test_features(50,2);
    // test that adding clusters works
    for(int i=1; i<51; i++)
    {
        string meanFileName = "../Data/mean" + boost::lexical_cast<std::string>(i) + ".mat";
        loadMatlabMatrix(meanFileName, "meanvar", mean_tmp);
        mean = mean_tmp.transpose();
        string covFileName = "../Data/cov" + boost::lexical_cast<std::string>(i) + ".mat";
        loadMatlabMatrix(covFileName, "covvar", cov);
        string nptsFileName = "../Data/npts" + boost::lexical_cast<std::string>(i) + ".mat";
        loadMatlabMatrix(nptsFileName, "ptsvar", npts_tmp);
        npts = npts_tmp;

        Cluster newCluster(mean, cov, npts(0,0));

        incremental_vocab.addCluster(newCluster);

        test_features.row(i-1) = mean;
    }
    std::cout << "Successfully read in test data\n";

    incremental_vocab.update();
    std::cout << "After update I've got " << incremental_vocab.size() << " words\n";
    incremental_vocab.setDesiredDimensionality(1);
    incremental_vocab.dimensionalityReduction();

    std::cout << "Trying another update\n";
    EigenMatrix proj1 = incremental_vocab.getCurrentProj();
    incremental_vocab.update();
    std::cout << "After update I've got " << incremental_vocab.size() << " words\n";
    
    EigenMatrix proj2 = incremental_vocab.getCurrentProj();

    EigenMatrix proj = proj2*proj1;

    int right = 0;
    int wrong = 0;
    for(int i=0; i < 50; i++)
    {
        EigenColVector temp = test_features.row(i).transpose();
        bool success = true;
        int res = incremental_vocab.clusterAssociation(temp,success);
        if(proj(res,i)!=1)
            wrong++;
        else
            right++;

    }

    std::cout << "Got " << right << " associations right  and " << wrong << " wrong\n";
    std::cout << "Within cluster ratio is " << incremental_vocab. withinClusterRatio() << "\n";


   std::ofstream ofs("Vocab_Save.dat");
    {
        boost::archive::text_oarchive oa(ofs);
        oa & incremental_vocab;
    }

    std::cout << "Wrote vocab OK\n";
    Vocab restored;
    std::ifstream ifs("Vocab_Save.dat");
    {
        boost::archive::text_iarchive ia(ifs);
        ia & restored;
    }
    std::cout << "Restored " << restored.size() << " visual words\n";
}
