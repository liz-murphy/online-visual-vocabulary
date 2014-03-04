#include <ovv/VisualWord.h>
#include <tree.hh>
#include <tree_util.hh>


VisualWord::VisualWord(Cluster &elemental_cluster)
{
    p_tree = new tree<Cluster>;
    p_tree->set_head(elemental_cluster);
    tree<Cluster>::iterator top = p_tree->begin();
    top->setID(s_nID++); 
}

VisualWord::~VisualWord()
{
   /* tree<Cluster>::post_order_iterator it = p_tree->begin_post();
    while(it != p_tree->end_post())
    {
        delete *it;
        ++it;
    }*/
    delete p_tree;
}

const EigenColVector &VisualWord::centroid()
{
    tree<Cluster>::iterator top = p_tree->begin();
    return top->mean_;
}

const EigenMatrix &VisualWord::covariance()
{
    tree<Cluster>::iterator top = p_tree->begin();
    return top->cov_;
}

int VisualWord::getID()
{
    tree<Cluster>::iterator top = p_tree->begin();
    return top->m_nID;
}

int VisualWord::npts()
{
    tree<Cluster>::iterator top = p_tree->begin();
    return top->npts_;
}

bool VisualWord::merge(VisualWord *merge_word)
{
    // create new root node
    double na = this->npts();
    double nb = merge_word->npts();
    EigenColVector Ca = this->centroid();
    EigenColVector Cb = merge_word->centroid();
    EigenMatrix Ra = this->covariance();
    EigenMatrix Rb = merge_word->covariance();

    EigenColVector mean = (na*Ca + nb*Cb)/(na+nb);
    EigenMatrix cov = (na-1)/(na+nb-1)*Ra + (nb-1)/(na+nb-1)*Rb + (nb*na)/((na+nb)*(na+nb-1))*((Ca-Cb)*(Ca-Cb).transpose());

    // add to the tree
    tree<Cluster>::iterator top = p_tree->begin();
    Cluster combined(mean, cov, na+nb); 
    p_tree->wrap(top, combined);
    p_tree->insert_subtree_after(top, merge_word->p_tree->begin());
    top = p_tree->begin();
    top->setID(s_nID++);

    return true;
}

size_type VisualWord::featureDistance(EigenColVector &feature, double stop_val)
{
    double min_val = std::numeric_limits<double>::infinity();
    tree<Cluster>::pre_order_iterator it = p_tree->begin();
    while(it != p_tree->end())
    {
        EigenColVector dist = it->centroid() - feature;
        EigenColVector eucdist = dist.rowwise().norm();
        if(eucdist(0,0) < min_val && it.number_of_children() == 0)
            min_val = eucdist(0,0);

        if(eucdist(0,0) > stop_val)
        {
            it.skip_children();
        }
        ++it;

    }
    return min_val;
}

bool VisualWord::prune(double p)
{
    bool res = false;
    tree<Cluster>::pre_order_iterator it = p_tree->begin();
    while(it != p_tree->end())
    {
        if(it->covariance().trace() < p*covariance().trace())
        {
            it=p_tree->erase(it);
            res = true;
        }
        else
            ++it;

    }
    return res;
}

bool VisualWord::doDimensionalityReduction(const EigenMatrix &ProjMat)
{
    tree<Cluster>::pre_order_iterator it = p_tree->begin();
    while(it != p_tree->end())
    {
        it->doDimensionalityReduction(ProjMat);
        ++it;
    }
    return true;  
}
