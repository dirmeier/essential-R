#include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::List dostuff()
{
  // some computations

  return Rcpp::List::create(
    Rcpp::Named("a") = 1
  );
}
