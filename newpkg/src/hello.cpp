/**
 * emptypRoject: description
 * <p>
 * Copyright (C) user
 * <p>
 * This file is part of emptypRoject.
 * <p>
 * emptypRoject is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * <p>
 * emptypRoject is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * <p>
 * You should have received a copy of the GNU General Public License
 * along with emptypRoject. If not, see <http://www.gnu.org/licenses/>.
 *
 */

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <vector>

//' Do something cool with your life
//'
//' @noRd
//' @param m  matrix
//' @return  returns also matrix
// [[Rcpp::interfaces(r, cpp)]]
// [[Rcpp::export(name=".hello.cpp")]]
Eigen::VectorXd mrwr_(const Eigen::MatrixXd& m)
{
  return m * m.transpose();
}