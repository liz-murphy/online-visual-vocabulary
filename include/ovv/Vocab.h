#ifndef VOCAB_H
#define VOCAB_H

#include <ovv/Cluster.h> 
#include <ovv/VisualWord.h> 

#include <boost/lexical_cast.hpp>
#include <matfile/Output>
#include <matfile/Wrappers/EigenDenseOWrapper.hpp>

#include <flann/flann.hpp>

typedef std::pair<int, int> clusterInd;
typedef std::pair<double, clusterInd> mergePair;

inline bool comparator(const mergePair &l, const mergePair &r)
{
    return l.first > r.first;
}

typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenBool;

extern "C" void dggev_(const char* JOBVL, const char* JOBVR, const int* N,
            const double* A, const int* LDA, const double* B, const int* LDB,
            double* ALPHAR, double* ALPHAI, double* BETA,
            double* VL, const int* LDVL, double* VR, const int* LDVR,
            double* WORK, const int* LWORK, int* INFO);
        
class Vocab
{
    public:
        Vocab();
        ~Vocab();
        bool update();  // update the vocabulary if association drops below 90% 
        bool associate(cv::Mat &feature); 
        bool addCluster(Cluster &elemental_cluster);
        bool merge(int w1, int w2);
        double computeMergeGain(int w1, int w2);
        int size();
        bool setDimensionality(int dim);
        bool setDesiredDimensionality(int dim){m_s=dim; return true;};
        bool setNN(int nn){m_nn = nn; return true;};
        bool updateKDTree();
        bool WriteToMatFiles();
        //bool clusterAssociation(const cv::Mat &dtors);
        int clusterAssociation(EigenColVector &feature, bool &success);
        const EigenMatrix &getCurrentProj();
        const EigenColVector &getIDF(){return m_idf;};
        double withinClusterRatio();
        bool prune();
        bool dimensionalityReduction();
        bool incrementIndexCount(){m_indexed++; return true;};
        int getDimensionality(){return m_Proj.cols();};
       EigenMatrix getProjMatrix(){return m_Proj;};
       EigenMatrix getSb(){return m_Sb;};
       EigenMatrix getSw(){return m_Sw;};
       const EigenColVector &getWordMean(int i){return m_clusters[i]->centroid();};
       const EigenMatrix &getWordCov(int i){return m_clusters[i]->covariance();};
   private:
        EigenMatrix m_Sw; // within clusters scatter matrix
        EigenMatrix m_Sb; // between clusters scatter matrix
        int id_to_index(int);
        friend class boost::serialization::access;
        template<class Archive>
        void save(Archive &ar, const unsigned int version) const
        {
            int nWords = m_clusters.size();
            ar & nWords;
            ar & m_Sw;
            ar & m_Sb;
            ar & m_C;
            ar & m_V;
            ar & m_N;
            ar & m_dim;
            ar & m_nn;
            ar & m_nVocab;
            ar & m_projections;
            ar & m_dataset;
            ar & m_features_seen;
            ar & m_update_criterion; 
            ar & bGenerateInit;
            ar & m_Proj;
            //ar & m_indexed;
            //ar & tf_idf;
            //ar & tf_idf_norms;
            //ar & m_idf;
            for(int i=0; i<nWords; i++)
                ar & m_clusters[i];
        };
        template<class Archive>
        void load(Archive &ar, const unsigned int version) 
        {
            int nWords;
            ar & nWords;
            ar & m_Sw;
            ar & m_Sb;
            ar & m_C;
            ar & m_V;
            ar & m_N;
            ar & m_dim;
            ar & m_nn;
            ar & m_nVocab;
            ar & m_projections;
            ar & m_dataset;
            ar & m_features_seen;
            ar & m_update_criterion; 
            ar & bGenerateInit;
            ar & m_Proj;
            //ar & m_indexed;
            //ar & tf_idf;
            //ar & tf_idf_norms;
            //ar & m_idf;
            for(int i=0; i<nWords; i++)
            {
                VisualWord *W = new VisualWord();
                ar & W;
                m_clusters.push_back(W);
            }
        };
        template<class Archive>
        void serialize(Archive &ar, const unsigned int version)
        {
            boost::serialization::split_member(ar, *this, version);
        }
        std::vector<VisualWord *> m_clusters;
        double m_N; // Lazy way to avoid perils of integer division
       EigenColVector m_C;  // global data centroid
        EigenColVector m_V;  // global data centroid
        int m_dim; // Dimensionality of the data
        int m_nn; // nearest neighbours to examine in vocab update step 
        int m_nVocab; // current version of Vocab
        double m_tau;
        std::vector<EigenMatrix> m_projections;
        EigenRowMajor m_dataset;
        //flann::Matrix<double> m_dataset;
        flann::Index<flann::L2<size_type> > *p_index; // kd-tree of visual word centres for use in online indexing
        double m_features_seen;
        double m_update_criterion;
        double m_p;
        EigenMatrix m_Proj;
        bool bGenerateInit;
        int m_s;    // number of desired dimensions of the data
        //int m_indexed;  // number of images indexed
        //std::vector<EigenColVector> tf_idf;
        //std::vector<double> tf_idf_norms;
        int m_lastVocabSize;
        EigenColVector m_idf;
        int m_indexed;
        std::vector<int> ids_;  // keep track of m_clusters to kdtree index relationship
        bool removed_;
        int size_;
        int last_id_;
        flann::DynamicBitset removed_points_;
        int removed_count_;
        flann::Matrix<double> *m_fDataset;
};
#endif
