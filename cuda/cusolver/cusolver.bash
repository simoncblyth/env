# === func-gen- : cuda/cusolver/cusolver fgp cuda/cusolver/cusolver.bash fgn cusolver fgh cuda/cusolver src base/func.bash
cusolver-source(){   echo ${BASH_SOURCE} ; }
cusolver-edir(){ echo $(dirname $(cusolver-source)) ; }
cusolver-ecd(){  cd $(cusolver-edir); }
cusolver-dir(){  echo $LOCAL_BASE/env/cuda/cusolver/cusolver ; }
cusolver-cd(){   cd $(cusolver-dir); }
cusolver-vi(){   vi $(cusolver-source) ; }
cusolver-env(){  elocal- ; }
cusolver-usage(){ cat << EOU



* https://stackoverflow.com/questions/28794010/solving-dense-linear-systems-ax-b-with-cuda

* https://devblogs.nvidia.com/parallel-direct-solvers-with-cusolver-batched-qr/

The cuSOLVER library provides factorizations and solver routines for dense and
sparse matrix formats, as well as a special re-factorization capability
optimized for solving many sparse systems with the same, known, sparsity
pattern and fill-in, but changing coefficients. A goal for cuSOLVER is to
provide some of the key features of LAPACK on the GPU, as users commonly
request LAPACK capabilities in CUDA libraries. cuSOLVER has three major
components: cuSolverDN, cuSolverSP and cuSolverRF, for Dense, Sparse and
Refactorization, respectively.

Let’s start with cuSolverDN, the dense factorization library. These are the
most like LAPACK, in fact cuSOLVER implements the LAPACK API with only minor
changes. cuSOLVER includes Cholesky factorization (potrf), LU factorization
(getrf), QR factorization (geqrf) and Bunch-Kaufmann LDL^{T} (symtrf), as well
as a GPU-accelerated triangular solve (getrs, potrs). For solving systems with
QR factorization, cuSOLVER provides ormqr to compute the orthogonal columns of
Q given A and R, and getrs to solve R. I’ll go into detail on these in a
followup post.

cuSolverSP provides sparse factorization and solve routines based on QR
factorization. QR can be used for solving linear systems and least-squares
problems. QR factorization is very robust, and unlike LU factorization, it
doesn’t rely on pivoting.

* https://en.wikipedia.org/wiki/Linear_least_squares

The cuSolverRF library can quickly update an existing LU factorization as the
coefficients of the matrix change. This has application in chemical kinetics,
combustion modeling and non-linear finite element methods. I’ll cover this more
in a followup post.

For this post, I’ll take a deep look into sparse direct factorization, using
the QR and batched QR features of cuSolverSP.




QR decomposition
~~~~~~~~~~~~~~~~~~~~

* https://en.wikipedia.org/wiki/QR_decomposition

In linear algebra, a QR decomposition (also called a QR factorization) of a
matrix is a decomposition of a matrix A into a product A = QR of an orthogonal
matrix Q and an upper triangular matrix R. QR decomposition is often used to
solve the linear least squares problem, and is the basis for a particular
eigenvalue algorithm, the QR algorithm.



Least Squares Problems

* http://www4.ncsu.edu/%7Emtchu/Teaching/Lectures/MA529/chapter4.pdf


* https://en.wikipedia.org/wiki/Levenberg–Marquardt_algorithm
  
  * solve non-linear least squares problems.


* http://on-demand.gputechconf.com/gtc/2015/posters/GTC_2015_Astronomy___Astrophysics_07_P5327_WEB.pdf




CUSP

* http://cusplibrary.github.io

  Cusp is a library for sparse linear algebra and graph computations based on
  Thrust. Cusp provides a flexible, high-level interface for manipulating sparse
  matrices and solving sparse linear systems.








Likelihood Review
---------------------


* https://onlinecourses.science.psu.edu/stat504/node/27/

  Whatever function of the parameter we get when we plug the observed data x into f(x ; θ), we call that function the likelihood function.

* https://stats.stackexchange.com/questions/112451/maximum-likelihood-estimation-mle-in-layman-terms



EOU
}
cusolver-get(){
   local dir=$(dirname $(cusolver-dir)) &&  mkdir -p $dir && cd $dir

}
