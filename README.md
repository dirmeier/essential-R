# Essential R

[![Project Status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)

:rocket: Essential tools and libraries for programming in R

## About

This document serves as a personal list of

* tools for package development,
* good practices for programming,
* and most frequently used packages

with an emphasis on data science, Bayesian stats and probabilistic ML.

## R package development

A minimum `R`-package stack at least consists of the following packages of tools:

- [`goodpractice`](https://github.com/MangoTheCat/goodpractice) for advice on writing R packages
- [`devtools`](https://github.com/r-lib/devtools) for general package development
- [`testthat`](https://github.com/r-lib/testthat) for unit testing
- [`roxygen2`](https://github.com/r-lib/testthat) for method documentation
- [`covr`](https://github.com/r-lib/covr) to generate coverage reports
- [`lintr`](https://github.com/jimhester/lintr) for static code analysis
- [`styler`](https://github.com/r-lib/styler) to automatically format code
- [`usethis`](https://github.com/r-lib/usethis) for automation of repetitive tasks during development
- [`remotes`](https://github.com/r-lib/remotes) to install R packages from Git repositories, CRAN and Bioconductor
- [`pkgdown`](https://github.com/r-lib/pkgdown) to generate websites of your package
- [`rcmdcheck`](https://github.com/r-lib/pkgdown) to check your package within R
- [`profvis`](https://github.com/rstudio/profvis) to visualize profiling data
- [`bench`](https://github.com/r-lib/bench) to time R expressions
- [`microbenchmark`](https://github.com/r-lib/bench) to also time R expressions
- [`lobstr`](https://github.com/r-lib/lobstr) to pry open R
- [`here`](https://github.com/r-lib/here) to find files and folders

Many more can be find on github at [r-lib](https://github.com/r-lib) or [rstudio](https://github.com/rstudio/).

## Programming

- [`purrr`](https://github.com/tidyverse/purrr) for functional programming
- [`magrittr`](https://github.com/tidyverse/magrittr) for pipeing function calls
- [`R6`](https://github.com/r-lib/R6) for object-oriented programming with encapsulation
- [`Rcpp`](https://github.com/RcppCore/Rcpp) (with `RcppArmadillo`, `RcppEigen` and `BH`) for integration of C++ code
- [`reticulate`](https://github.com/rstudio/reticulate) for interfacing to Python
- [`cpp11`](https://github.com/r-lib/cpp11) as alternative (and complement) to Rcpp
- `compiler`, `doParallel`, `parallel`, `foreach` for speeding up R
- [`rlang`](https://github.com/dirmeier/datastructures) as low-level API for programming in R

### C++

- [`meson`](https://mesonbuild.com/) as modern build system and [`autotools`](https://www.gnu.org/software/automake/manual/html_node/Autotools-Introduction.html) since R is old
- [`gdb`](https://www.gnu.org/software/gdb/) and [`lldb`](https://lldb.llvm.org/) for debugging C++
- [`sanitizers`](https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html) to detect memory leaks, etc.
- [`cppcheck`](http://cppcheck.sourceforge.net/) for static code analysis
- [`cpplint`](https://github.com/cpplint/cpplint) to check C++ style
- [`clang-format`](https://clang.llvm.org/docs/ClangFormat.html) to format C++ files
- [`doxygen`](http://cppcheck.sourceforge.net/) for code documentation
- [`boost`](https://www.boost.org/) for unit tests, data structures and basically everything you ever need

Furthermore, some good reading:

- Scott Meyers: Effective C++
- Scott Meyers: Effective Modern C++
- Scott Meyers: Effective STL
- Kurt Guntheroth: Optimized C++
- David Vandevoorde: C++ Templates - the compete guide
- Nicolai Josuttis: C++17 - the complete guide

## Working with data

- [`tidyverse`](https://github.com/tidyverse/tidyverse) (`dplyr`, `tidyr`, ...) for working with data in general
- [`data.table`](https://github.com/Rdatatable/data.table) as a fast alternative to R`s native data frame
- [`datastructures`](https://github.com/dirmeier/datastructures) for advanced data structures
- [`dbplyr`](https://github.com/tidyverse/dbplyr/) to work with data bases
- [`RSQLite`](https://db.rstudio.com/databases/sqlite/) to work with SQLite data bases

## Machine learning and Bayesian statistics 

- [`rstan`](https://github.com/stan-dev/rstan) to fit Bayesian models
- [`cmdstanr`](https://github.com/stan-dev/cmdstanr) as light-weight interface to `cmdstan`
- [`rstanarm`](https://github.com/stan-dev/rstanarm) for applied regression modeling
- [`brms`](https://github.com/paul-buerkner/brms) for  multilevel models
- [`bayesplot`](https://github.com/stan-dev/bayesplot) to visualize Bayesian inferences
- [`loo`](https://github.com/stan-dev/loo) for approximate LOO-CV and PSIS
- [`projpred`](https://github.com/stan-dev/projpred) for projection predictive variable selection 
- [`rstantools`](https://github.com/stan-dev/rstantools) for developing R Packages interfacing with Stan 
- [`posterior`](https://github.com/stan-dev/posterior) for working with output from Bayesian models
- [`coda`](https://cran.r-project.org/web/packages/coda/index.html) for summarizing and working with MCMC output
- [`MCMCpack`](https://cran.r-project.org/web/packages/MCMCpack/index.html) provides some utility functions
- [`LaplacesDemon`](https://github.com/LaplacesDemonR/LaplacesDemon) for even more Bayes utility
- [`mgcv`](https://cran.r-project.org/web/packages/mgcv/index.html) for generalized additive mixed models
- [`tensorflow`](https://github.com/rstudio/tensorflow) for numerical computation
- [`tfprobability`](https://cran.r-project.org/web/packages/mgcv/index.html) for statistical computation and probabilistic modeling
- [`keras`](https://github.com/rstudio/tensorflow) to work with neural networks
- [`sparklyr`](https://github.com/sparklyr/sparklyr) for big data processing
- [`kernlab`](https://cran.r-project.org/web/packages/kernlab/index.html) for kernel-based machine learning

## Graphical models and causal inference

- [`bnlearn`](https://cran.r-project.org/web/packages/bnlearn/) for BN structure learning
- [`pcalg`](https://cran.r-project.org/web/packages/pcalg/) for causal inference using graphical models
- [`ggdag`](https://github.com/malcolmbarrett/ggdag) for visualizing DAGs
- [`dagitty`](https://github.com/jtextor/dagitty) for analysis of structural equation models

## Optimization

- [`nloptr`](https://cran.r-project.org/web/packages/nloptr/index.html) for non-linear optimization
- [`cvxr`](https://github.com/cvxgrp/CVXR) for disciplined convec optimization

## Visualization

- [`ggplot2`](https://github.com/tidyverse/ggplot2) as base package for visualization
- [`paletteer`](https://github.com/EmilHvitfeldt/paletteer) for all the palettes
- [`vapoRwave`](https://github.com/moldach/vapoRwave) for even more palettes
- [`gpubr`](https://github.com/kassambara/ggpubr) for publication ready visualization
- [`ggthemes`](https://github.com/jrnold/ggthemes) for additional themes and scales for ggplot2
- [`hrbrthemes`](https://github.com/hrbrmstr/hrbrthemes) for even more themes
- [`ggsci`](https://github.com/nanxstats/ggsci) for sci-fi themes
- [`ggthemr`](https://github.com/cttobin/ggthemr) for even mroe themes
- [`colourlovers`](https://github.com/andrewheiss/colourlovers) for access to the COLOURlovers API 
- [`patchwork`](https://github.com/thomasp85/patchwork) to easily compose plots
- [`colorspace`](https://cran.r-project.org/web/packages/colorspace/index.html) for color manipulation
- [`cowplot`](https://github.com/wilkelab/cowplot) for plot annotations
- [`scales`](https://github.com/r-lib/scales) and [`swatches`](https://github.com/hrbrmstr/swatches) 
- [`plotly`](https://github.com/ropensci/plotly) for interactive plots
- [`ggraph`](https://github.com/thomasp85/ggraph) to visualize graphs
- [`gganimate`](https://github.com/thomasp85/gganimate) to animate plots
- [`DiagrammeR`](https://github.com/rich-iannone/DiagrammeR) for diagrammes, graphs and networks
- [`highcharter`](https://github.com/jbkunst/highcharter) as alternative to `plotly`
- [`tidybayes`](https://github.com/mjskay/tidybayes) for geoms for Bayesian models
- [`scico`](https://github.com/thomasp85/scico) for more color palettes
- [`ggnetwork`](https://github.com/briatte/ggnetwork) for geoms for networks
- [`tweenr`](https://github.com/thomasp85/tweenr) to interpolate data
- [`magick`](https://github.com/ropensci/magick) to work with images
- [`imager`](https://github.com/dahtah/imager) as complement to `magick`
- [`r2d3`](https://github.com/rstudio/r2d3) to interface to [`d3`](https://d3js.org/)

# Reporting

- [`shiny`](https://github.com/rstudio/shiny) for interactive web applications
- [`rmarkdown`](https://rmarkdown.rstudio.com/index.html) to generate websites, pdf documents, etc from markdown files
- [`bookdown`](https://github.com/rstudio/bookdown) for authoring books
- [`tufte`](https://github.com/rstudio/tufte) for Tufte-style documents
- [`xaringan`](https://github.com/yihui/xaringan) for HTML presentations

## Other

- [`igraph`](https://github.com/igraph/igraph) to work with graphs
- [`drake`](https://github.com/ropensci/drake) for building pipelines

## Author

Simon Dirmeier <a href="mailto:simon.dirmeier@web.de">simon.dirmeier@web.de</a>

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work by Simon Dirmeier is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
