#include <ovv/Vocab.h>
#include <algorithm>
#include <ovv/UniqueNumber.h>
#include <ovv/EigenIndexVec.h>
#include <ovv/EigenIndexVecColMajor.h>
#include <ovv/IndexVec.h>
#include <boost/bind.hpp>
#include <opencv2/core/eigen.hpp>
#include <flann/algorithms/kdtree_index.h>
#include <flann/flann.hpp>
#include <flann/flann.h>
Vocab::Vocab():m_nn(50), m_tau(1.4), m_p(0.1), bGenerateInit(true), m_dim(128) 
{
    m_nVocab = 0;
    p_index = NULL;
    m_features_seen = 0;
    m_update_criterion = 0;
    m_Proj = EigenMatrix::Identity(m_dim, m_dim);
    m_idf = EigenColVector::Zero(2,1);
    m_s = m_dim;
    m_lastVocabSize = 0;
    m_indexed = 0;
}

bool Vocab::setDimensionality(int dim)
{
    m_dim = dim;
    m_Proj = EigenMatrix::Identity(m_dim, m_dim);
    return true;
}

Vocab::~Vocab()
{
    std::vector<VisualWord *>::iterator it;
    for(it = m_clusters.begin(); it != m_clusters.end(); )
    {
        delete *it;
        it = m_clusters.erase(it);
    }
    delete p_index;
}


int Vocab::size()
{
    return m_clusters.size(); 
}

/* I have unit tested this one */

bool Vocab::addCluster(Cluster &elemental_cluster)
{
    // update within-cluster scatter matrix
    double ne = elemental_cluster.npts();

    // Do dimensionality reduction
    if(elemental_cluster.centroid().rows() != m_s)
        elemental_cluster.doDimensionalityReduction(m_Proj);

    if(m_Sw.cols() == 0)
    {
        int r = elemental_cluster.covariance().rows();
        int r1 = elemental_cluster.centroid().rows();
        int c = elemental_cluster.covariance().cols();
        int c1 = elemental_cluster.centroid().cols();

        m_N = 0;
        m_Sw.resize(r,c);
        m_C.resize(r1,c1);
        m_Sb.resize(r,c);
        m_V.resize(r1,c1);


        m_Sw = elemental_cluster.covariance(); 
        m_C = elemental_cluster.centroid();
        m_Sb = EigenMatrix::Zero(r,c);
        m_V = EigenColVector::Zero(r1,c1);
        m_N = ne;
    }
    else
    {

        m_Sw = (m_N*m_Sw + ne*elemental_cluster.covariance())/(m_N+ne);

        EigenColVector C_dash = (m_C*m_N + elemental_cluster.centroid()*ne)/(m_N + ne);

        EigenColVector delta_C = C_dash - m_C;

        EigenColVector V_dash = (m_N*m_V + m_N*delta_C + ne*(elemental_cluster.centroid()-C_dash))/(m_N + ne);

        m_Sb =  m_N/(m_N+ne)*(m_Sb - delta_C*delta_C.transpose() + V_dash*delta_C.transpose() + delta_C*V_dash.transpose()) + ne/(ne+m_N)*(elemental_cluster.centroid()-C_dash)*(elemental_cluster.centroid()-C_dash).transpose();

        m_C = C_dash;

        m_V = V_dash;
        
        m_N = m_N + ne;
    }

    VisualWord *newCluster;
    try
    {
        newCluster = new VisualWord(elemental_cluster);
    }
    catch(const std::bad_alloc&)
    {
        std::cout << "Can't allocate new visual word, out of memory\n";
        exit(0);
    }

    m_clusters.push_back(newCluster);

    return true;
}

/* Unit tested this, working OK */

double Vocab::computeMergeGain(int w1, int w2)
{
    double na = m_clusters[w1]->npts();
    EigenColVector Ca = m_clusters[w1]->centroid();
    EigenMatrix Ra = m_clusters[w1]->covariance();

    double nb = m_clusters[w2]->npts();
    EigenColVector Cb = m_clusters[w2]->centroid();
    EigenMatrix Rb = m_clusters[w2]->covariance();

    EigenColVector Cab= (na*Ca + nb*Cb)/(na+nb);

    EigenMatrix Rab= (na-1)/(na+nb-1)*Ra + (nb-1)/(na+nb-1)*Rb + (nb*na)/((na+nb)*(na+nb-1))*((Ca-Cb)*(Ca-Cb).transpose());
  
    int pts = na+nb;
    
    EigenMatrix Sb_dash = m_Sb + pts/m_N*(m_C-Cab)*(m_C-Cab).transpose() - na/m_N*(m_C-Ca)*(m_C-Ca).transpose() - nb/m_N*(m_C-Cb)*(m_C-Cb).transpose();
     
    EigenMatrix Sw_dash = m_Sw + pts/m_N*Rab - na/m_N*Ra - nb/m_N*Rb;
   
    double Q = m_Sb.trace()/(m_Sw.trace());
   
    double Q_dash = Sb_dash.trace()/(Sw_dash.trace());

    return(Q_dash - Q); 
}

bool Vocab::merge(int w1, int w2)
{
    double na = m_clusters[w1]->npts();
    EigenColVector Ca = m_clusters[w1]->centroid();
    EigenMatrix Ra = m_clusters[w1]->covariance();

    double nb = m_clusters[w2]->npts();
    EigenColVector Cb = m_clusters[w2]->centroid();
    EigenMatrix Rb = m_clusters[w2]->covariance();

    m_clusters[w1]->merge(m_clusters[w2]);

    double nab = m_clusters[w1]->npts();
    EigenColVector Cab = m_clusters[w1]->centroid();
    EigenMatrix Rab = m_clusters[w1]->covariance();
   
    m_Sb = m_Sb + nab/m_N*(m_C-Cab)*(m_C-Cab).transpose() - na/m_N*(m_C-Ca)*(m_C-Ca).transpose() - nb/m_N*(m_C-Cb)*(m_C-Cb).transpose();
     
    m_Sw = m_Sw + nab/m_N*Rab - na/m_N*Ra - nb/m_N*Rb;
    
    return true;
}

bool Vocab::updateKDTree()
{
    if(p_index != NULL)
    {
        //delete[] m_fDataset->ptr();
        delete p_index;
    }
 
    if(bGenerateInit)
        m_dataset.resize(m_clusters.size(), m_dim);
    else 
        m_dataset.resize(m_clusters.size(), m_s);
    
    for(unsigned int i=0; i < m_clusters.size(); i++)
    {
        m_dataset.row(i) = m_clusters[i]->centroid().transpose();
    }

    //std::copy(m_dataset.data(), m_dataset.data()+m_dataset.size(), *m_fDataset);
    
    flann::Matrix<size_type> input(m_dataset.data(), m_dataset.rows(), m_dataset.cols());
    try
    {
        p_index = new flann::Index<flann::L2<size_type> >(input, flann::KDTreeIndexParams(4));
    }
    catch(const std::bad_alloc&)
    {
        std::cout << "Can't create new index, out of memory\n";
        exit(0);
    }
 
    // construct a randomized kd-tree index using 4 kd-trees
    p_index->buildIndex();

   return true;
}

bool Vocab::update()
{

    std::cout << "***********Starting update with " << m_clusters.size() << " words\n";
    if(m_lastVocabSize==0)
        m_lastVocabSize = m_clusters.size();

    EigenBool transformation = EigenBool::Identity(m_clusters.size(),m_lastVocabSize);

    // update the kd tree to take into account elementary clusters that will have been added
    updateKDTree();

    // store results in here 
    // std::vector<std::vector <int> >indices;
    // std::vector<std::vector <double> >dists;

    // do a knn search, using 128 checks
    flann::Matrix<size_type> fQuery(m_dataset.data(), m_dataset.rows(), m_dataset.cols());
    //flann::Matrix<double> fQuery = *m_fDataset;
    flann::Matrix<int> fIndices(new int[fQuery.rows*m_nn], fQuery.rows, m_nn);
    flann::Matrix<size_type> fDists(new size_type[fQuery.rows*m_nn], fQuery.rows, m_nn);
    p_index->knnSearch(fQuery, fIndices, fDists, m_nn, flann::SearchParams(128));
  
    // make an ordered list of merging candidates
    std::vector<mergePair> merge_candidates;

    for(unsigned int i=0; i < fIndices.rows; i++)
    {
        for(unsigned int j=0; j < fIndices.cols; j++) 
        {
            int other_ind = fIndices[i][j];
            if(other_ind <= (int)i)
                continue;
            double res = computeMergeGain(i,other_ind);
            if(res > 0)
                merge_candidates.push_back(mergePair(res,clusterInd(i,other_ind)));
        }
    }

    delete[] fIndices.ptr();
    delete[] fDists.ptr();

    sort(merge_candidates.begin(),merge_candidates.end(), comparator);

    // Set up a way of tracking potential merge candidates through earlier merges
    std::vector<int> index_tracker(m_clusters.size(),-1);

    for(unsigned int i = 0; i < merge_candidates.size(); i++)
    {
        int i1 = merge_candidates[i].second.first;
        int i2 = merge_candidates[i].second.second;
        
        while(index_tracker[i1] != -1)
            i1 = index_tracker[i1];

        while(index_tracker[i2] != -1)
            i2 = index_tracker[i2];

        if(i1==i2)
            continue;

        merge(i1,i2);

        transformation.row(i1)+=transformation.row(i2);

        index_tracker[i2]=i1;
    }

    for(int i=index_tracker.size()-1; i>=0; i--)
    {
        if(index_tracker[i] != -1)
        {
            delete m_clusters[i];  // try this
            m_clusters.erase(m_clusters.begin()+i);
            // do bookkeeping on ids
        }
    }

    EigenBool proj = EigenBool::Zero(m_clusters.size(), m_lastVocabSize);    // One row for each of the new clusters
    // Create the projection matrix
    int projrow = 0;
    for(unsigned int i=0; i < index_tracker.size(); i++)
    {
        if(index_tracker[i]==-1)    // it hasn't been merged
        {
            proj.row(projrow) = transformation.row(i);
            projrow++;
        }
    }
    
    EigenMatrix iProj = proj.cast<size_type>();
    m_projections.push_back(iProj);
    prune();
       
    updateKDTree(); 
    m_features_seen = 0; 
    m_update_criterion = 0;

    if(bGenerateInit)
        bGenerateInit = false; 

   /* m_idf.resize(m_clusters.size(),1);
    for(unsigned int i=0; i < m_clusters.size(); i++)
     {
        m_idf(i) = (double)m_indexed/(double)m_clusters[i]->npts(); 
        if(m_idf(i) == 0)
        {
            std::cout << "ERROR, got zero in idf term, m_indexed is " <<  m_indexed << "\n";
            exit(0);
        }
     }

    m_idf = m_idf.array().log();
*/
    /*for(unsigned int i=0; i<tf_idf.size(); i++)
    {
        tf_idf[i] = iProj*tf_idf[i];
    }*/
    m_lastVocabSize = m_clusters.size();
    return true;

}

bool Vocab::WriteToMatFiles()
{
    for(unsigned int i=1; i < m_clusters.size()+1; i++)
    {
        std::string mean_var_name = "Vocab" + boost::lexical_cast<std::string>(m_nVocab) + "Word" +boost::lexical_cast<std::string>(i) + "Mean";
        std::string mean_file_name = mean_var_name + ".mat";
        matfile::OutputDevice mean_file(mean_file_name);

        // Create output wrapper for the matrix
        matfile::EigenDenseOWrapper wrapperDense(m_clusters[i-1]->centroid());
        mean_file.writeArray(mean_var_name, wrapperDense);
        
        std::string cov_var_name = "Vocab" + boost::lexical_cast<std::string>(m_nVocab) + "Word" +boost::lexical_cast<std::string>(i) + "Cov";
        std::string cov_file_name = cov_var_name + ".mat";
        matfile::OutputDevice cov_file(cov_file_name);

        // Create output wrapper for the matrix
        matfile::EigenDenseOWrapper wrapperDense2(m_clusters[i-1]->covariance());
        cov_file.writeArray(cov_var_name, wrapperDense2);
    }

    std::string mSb_var_name = "Sb";
    std::string mSb_file_name = mSb_var_name + ".mat";
    matfile::OutputDevice mSb_file(mSb_file_name);

    matfile::EigenDenseOWrapper wrapperDense3(m_Sb);
    mSb_file.writeArray(mSb_var_name, wrapperDense3);

    std::string mSw_var_name = "Sw";
    std::string mSw_file_name = mSw_var_name + ".mat";
    matfile::OutputDevice mSw_file(mSw_file_name);

    matfile::EigenDenseOWrapper wrapperDense4(m_Sw);
    mSw_file.writeArray(mSw_var_name, wrapperDense4);


    return true;
}

/*bool Vocab::clusterAssociation(const cv::Mat &dtors)
{
    Eigen::MatrixXd featureMat;
    cv::cv2eigen(dtors, featureMat);  // each row is a feature

    EigenMatrix featureMatProj = featureMat.transpose(); 
    featureMatProj = m_Proj.transpose()*featureMatProj;  // each column is a projected feature
    
    // generate a response histogram
    EigenColVector tf = EigenColVector::Zero(m_clusters.size(), 1);

    for(int i=0; i < featureMatProj.cols(); i++)
    {
        EigenColVector tempCol = featureMatProj.col(i);
        int word = clusterAssociation(tempCol);
        tf(word) = tf(word)+1;
    }
    tf = tf.array()/tf.sum();
   
    //EigenVector w = tf);//*m_idf.array(); 
    tf_idf.push_back(tf);
    //double wNorm2 = w.squaredNorm();
    //tf_idf_norms.push_back(wNorm2);

    EigenColVector Wr = tf.array()*m_idf.array().log();
    double WrNorm2 = Wr.squaredNorm();
    // Do comparison with all indexed images
    for(int i=0; i < ((int)tf_idf.size()-100); i++)
    {
        EigenColVector Wq = tf_idf[i].array()*m_idf.array();
        double WqNorm2 = Wq.squaredNorm();

        Eigen::MatrixXd Srq = (Wr.transpose()*Wq)/(WrNorm2*WqNorm2);
        if(Srq(0,0) > 0.45)
            std::cout << "Image " << tf_idf.size()-1 << " matches image " << i << " score is: " << Srq << "\n";

    }
    return true;
}
*/
int Vocab::clusterAssociation(EigenColVector &feature, bool &success)
{
    // find distance between feature and all cluster centroids
    if(feature.rows() != m_s)
        feature = m_Proj.transpose()*feature;
    // store results in here 
    //std::vector<std::vector <int> >indices;
    //std::vector<std::vector <double> >dists;

    EigenRowMajor featureRowMajor = feature.transpose();
    
    flann::Matrix<size_type> fQuery(featureRowMajor.data(), featureRowMajor.rows(), featureRowMajor.cols()); 
    flann::Matrix<int> fIndices(new int[fQuery.rows*m_nn], fQuery.rows, m_nn);
    flann::Matrix<size_type> fDists(new size_type[fQuery.rows*m_nn], fQuery.rows, m_nn);

    // do a knn search, using 128 checks
    //flann::Matrix<double> input(featureRowMajor.data(), featureRowMajor.rows(), featureRowMajor.cols());
 
    p_index->knnSearch(fQuery, fIndices, fDists, m_nn, flann::SearchParams(128));
  
    // indices and dists now holds sorted indexes in order of distance from feature

    std::vector<int> trees_to_visit;

    size_type min_dist = fDists[0][0]; 
    for(unsigned int i=0; i < fDists.cols; i++)
    {
        if(fDists[0][i] > m_tau*min_dist)
            break;
        trees_to_visit.push_back(fIndices[0][i]);
    }

    //delete[] fQuery.ptr();
    delete[] fIndices.ptr();
    delete[] fDists.ptr();

    std::vector<size_type> vword_distance(trees_to_visit.size());  // store the minimum distance feature from each tree
#pragma omp parallel 
{
#pragma omp for
    // Now visit the trees 
    for(unsigned int i=0; i < trees_to_visit.size(); i++)
    {
        vword_distance[i] = m_clusters[trees_to_visit[i]]->featureDistance(feature, min_dist*m_tau);    
    }
}
    // what is the closest visual word?
    std::vector<int> ind_vector2(vword_distance.size());
    std::generate(ind_vector2.begin(), ind_vector2.end(), UniqueNumber());

    IndexVec ind2(vword_distance);

    sort(ind_vector2.begin(),ind_vector2.end(),boost::bind(&IndexVec::comp,&ind2, _1, _2));
    
    int closest_word = trees_to_visit[ind_vector2[0]];

    EigenArray stddevs = 1.0*(m_clusters[closest_word]->covariance().diagonal().array().sqrt());
    EigenColVector diff = feature - m_clusters[closest_word]->centroid();
    EigenArray diff2 = diff.array().abs();//.cwiseAbs().array();
    if(!((diff2 <= stddevs).any()) )
    {
        /*std::string feature_var = "Feature" + boost::lexical_cast<std::string>(m_update_criterion);
        std::string feature_file_name = feature_var + ".mat";
        matfile::OutputDevice feature_file(feature_file_name);
        matfile::EigenDenseOWrapper wrapperDense(feature); 
        feature_file.writeArray(feature_var, wrapperDense);

        std::string cluster_var = "ClusterMean" + boost::lexical_cast<std::string>(m_update_criterion);
        std::string cluster_file_name = cluster_var + ".mat";
        matfile::OutputDevice cluster_file(cluster_file_name);
        matfile::EigenDenseOWrapper wrapperDense2(m_clusters[closest_word]->centroid()); 
        cluster_file.writeArray(cluster_var, wrapperDense2);

        std::string cov_var = "ClusterCovariance" + boost::lexical_cast<std::string>(m_update_criterion);
        std::string cov_file_name = cov_var + ".mat";
        matfile::OutputDevice cov_file(cov_file_name);
        matfile::EigenDenseOWrapper wrapperDense3(m_clusters[closest_word]->covariance()); 
        cluster_file.writeArray(cov_var, wrapperDense3);
*/
        m_update_criterion++; 
        success = false;
    }
    m_features_seen++;
   
    return closest_word;
}

const EigenMatrix &Vocab::getCurrentProj()
{
    return m_projections[m_projections.size()-1];
}

double Vocab::withinClusterRatio()
{
    if(m_features_seen == 0)
        return 1.0;
    else
        return (m_features_seen - m_update_criterion)/m_features_seen;
}

bool Vocab::prune()
{
    bool res = false;
    for(unsigned int i=0; i<m_clusters.size(); i++)
    {
        res = res || m_clusters[i]->prune(m_p);
    }
    return res;
}

bool Vocab::dimensionalityReduction()
{
    // Solving the Generalized Eigen-Problem
    // A*v(j) = lambda(j)*B*v(j)
    // with:
    // S_b*w = lambda*S_w*w
    // .... want to solve for w (eigen values) and use projection provided by eigen vectors of the n highest values of w

    Eigen::MatrixXd A = m_Sb.cast<double>();
    Eigen::MatrixXd B = m_Sw.cast<double>();
    Eigen::MatrixXd v, lambda;

    int N = A.cols(); // Number of columns of A and B. Number of rows of v.
    if (B.cols() != N  || A.rows()!=N || B.rows()!=N)
        return false;

    v.resize(N,N);
    lambda.resize(N, 3);

    int LDA = A.outerStride();
    int LDB = B.outerStride();
    int LDV = v.outerStride();

    double WORKDUMMY;
    int LWORK = -1; // Request optimum work size.
    int INFO = 0;
                                  
    double * alphar = const_cast<double *>(lambda.col(0).data());
    double * alphai = const_cast<double *>(lambda.col(1).data());
    double * beta   = const_cast<double *>(lambda.col(2).data());

    // Get the optimum work size.
    dggev_("N", "V", &N, A.data(), &LDA, B.data(), &LDB, alphar, alphai, beta, 0, &LDV, v.data(), &LDV, &WORKDUMMY, &LWORK, &INFO);

    LWORK = int(WORKDUMMY) + 32;
    Eigen::VectorXd WORK(LWORK);

    dggev_("N", "V", &N, A.data(), &LDA, B.data(), &LDB, alphar, alphai, beta, 0, &LDV, v.data(), &LDV, WORK.data(), &LWORK, &INFO);

    if(INFO==0)
    {
        // All good
        std::vector<int> ind_vector(v.rows());
        std::generate(ind_vector.begin(), ind_vector.end(), UniqueNumber());

        Eigen::MatrixXd arr1 = lambda.col(0);
        Eigen::MatrixXd arr2 = lambda.col(2);
    
        Eigen::MatrixXd eigenvalues = (arr1.array()/arr2.array()).abs(); // right eigenvalues
        EigenIndexVecColMajor ind1(eigenvalues);

        sort(ind_vector.begin(),ind_vector.end(),boost::bind(&EigenIndexVecColMajor::gt,&ind1, _1, _2));

        // Create a projection matrix
        //EigenMatrix Proj(m_dim, m_s);

        m_Proj.resize(m_dim, m_s);
        for(int i=0; i < m_s; i++)
        {
            m_Proj.col(i) = v.cast<size_type>().col(ind_vector[i]);
        }

        //m_Proj = Proj.transposeInplace;  // should do the conversion between RowMajor and ColMajor????
       
        for(unsigned int i=0; i < m_clusters.size(); i++)
            m_clusters[i]->doDimensionalityReduction(m_Proj);

        m_Sw = m_Proj.transpose()*m_Sw*m_Proj;
        m_Sb = m_Proj.transpose()*m_Sb*m_Proj;
        //m_dataset = m_dataset*m_Proj;

        m_V = m_Proj.transpose()*m_V;
        m_C = m_Proj.transpose()*m_C;

        updateKDTree();

        return true;
    }
    else
      return false;
}

int Vocab::id_to_index(int id)
{
    if(ids_.size()==0)
    {
        return id;
    }

    int cluster_index = -1;
    if(ids_[id] == id)
        return id;
    else
    {
        // binary search
        int start = 0;
        int end = ids_.size();
        while(start < end)
        {
            int mid = (start+end)/2;
            if(ids_[mid]==id)
            {
                cluster_index = mid;
                break;
            }
        }
    }
    return cluster_index;
}
