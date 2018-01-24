// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

// [[Rcpp::export]]
Eigen::MatrixXd square(Eigen::MatrixXd& m)
{
    return m.transpose() * m;
}
