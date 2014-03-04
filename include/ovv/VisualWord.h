#ifndef VISUAL_WORD_H
#define VISUAL_WORD_H

#include <ovv/tree.hh>
//#include <tree_util.hh>
#include <ovv/Cluster.h>
#include <opencv2/core/core.hpp>
#include <ovv/SerializationUtils.h>
#include <boost/serialization/split_member.hpp>
#include <iostream>

class VisualWord 
{
    public:
        VisualWord(){}; 
        ~VisualWord(); 
        VisualWord(Cluster &elemental_cluster); 
        bool merge(VisualWord *merge_word);
        bool associate(cv::Mat &feature); // does it fall witin radius?
        const EigenColVector &centroid();
        const EigenMatrix &covariance();
        int npts();
        int getID();
 //       void printTree(){kptree::print_tree_bracketed(*p_tree, std::cout);};
        size_type featureDistance(EigenColVector &feature, double stop_val);
        bool prune(double p);
        friend class Vocab;
        bool doDimensionalityReduction(const EigenMatrix &Projection);
    private:
        friend class boost::serialization::access;
/*        template<class Archive>
        void serialize(Archive &ar, const unsigned int version)
        {
            ar & s_nID;
            ar & p_tree;
        };*/
        template<class Archive>
        void save(Archive &ar, const unsigned int version) const
        {
            ar & s_nID;
            ar & p_tree;
        }
        template<class Archive>
        void load(Archive &ar, const unsigned int version)
        {
            ar & s_nID;
            p_tree = new tree<Cluster>;
            ar & p_tree;
        }
        template<class Archive>
        void serialize(Archive &ar, const unsigned int version)
        {
            boost::serialization::split_member(ar,*this,version);
        }
       // BOOST_SERIALIZATION_SPLIT_MEMBER()
        tree<Cluster> *p_tree;   // tree of elemental clusters
        static int s_nID;
};

#endif
