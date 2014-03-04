#ifndef SERIALIZATION_UTILS
#define SERIALIZATION_UTILS
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>
#include <ovv/Cluster.h>
#include <ovv/tree.hh>

namespace boost {
namespace serialization {

template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
inline void serialize(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> &t, const unsigned int file_version)
{
    size_t rows = t.rows(), cols = t.cols();
    ar & rows;
    ar & cols;
    if( rows * cols != (size_t)t.size() )
    t.resize( rows, cols );

    for(size_t i=0; i<(size_t)t.size(); i++)
    ar & t.data()[i];
}

template<class Archive, typename T>
void serialize(Archive &ar, tree<T> &t, const unsigned int file_version)
{
    ar & t.head;
    ar & t.feet;
}

template<class Archive, typename T>
void serialize(Archive &ar, tree_node_<T> &t, const unsigned int file_version)
{
            ar & t.parent;
            ar & t.first_child;
            ar & t.last_child;
            ar & t.next_sibling;
            ar & t.prev_sibling;
            ar & t.data;
}

}
}
#endif
