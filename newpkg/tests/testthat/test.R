# emptypRoject: description
#
# Copyright (C) user
#
# This file is part of emptypRoject.
#
# emptypRoject is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# emptypRoject is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with emptypRoject. If not, see <http://www.gnu.org/licenses/>.


context("hello")


testthat::test_that("i know my math", {
    testthat::equal("welcome to R-bones\n", f())
})

testthat::test_that("i know my math", {
    testthat::expect_false("wrong" == g())
})

if (requireNamespace("lintr", quietly = TRUE)) {
  testthat::test_that("package has style", {
    lintr::expect_lint_free()
  })
}