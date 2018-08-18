# === func-gen- : fit/gpufit fgp fit/gpufit.bash fgn gpufit fgh fit
gpufit-src(){      echo fit/gpufit.bash ; }
gpufit-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gpufit-src)} ; }
gpufit-vi(){       vi $(gpufit-source) ; }
gpufit-env(){      elocal- ; }
gpufit-usage(){ cat << EOU


* :google:`CUDA maximum likelihood estimation`

* :google:`Levenberg-Marquardt maximum likelihood minimization`

* https://github.com/topics/non-linear-regression



* https://www.researchgate.net/publication/258593416_Monte_Carlo-based_Reconstruction_in_Water_Cherenkov_Detectors_using_Chroma

Monte Carlo-based Reconstruction in Water Cherenkov Detectors using Chroma
March 2012
Stanley Seibert
Anthony Latorre

We demonstrate the feasibility of event reconstruction---including position,
direction, energy and particle identification---in water Cherenkov detectors
with a purely Monte Carlo-based method. Using a fast optical Monte Carlo
package we have written, called Chroma, in combination with several variance
reduction techniques, we can estimate the value of a likelihood function for an
arbitrary event hypothesis. The likelihood can then be maximized over the
parameter space of interest using a form of gradient descent designed for
stochastic functions. Although slower than more traditional reconstruction
algorithms, this completely Monte Carlo-based technique is universal and can be
applied to a detector of any size or shape, which is a major advantage during
the design phase of an experiment. As a specific example, we focus on
reconstruction results from a simulation of the 200 kiloton water Cherenkov far
detector option for LBNE.




* https://github.com/gpufit/GPUfit
* https://gpufit.readthedocs.io/en/latest/
* https://www.nature.com/articles/s41598-017-15313-9

GPU-accelerated Levenberg-Marquardt curve fitting in CUDA


* https://en.wikipedia.org/wiki/Cholesky_decomposition

Non-linear optimization[edit]

Non-linear multi-variate functions may be minimized over their parameters using
variants of Newton's method called quasi-Newton methods. At each iteration, the
search takes a step s defined by solving Hs = −g for s, where s is the step, g
is the gradient vector of the function's partial first derivatives with respect
to the parameters, and H is an approximation to the Hessian matrix of partial
second derivatives formed by repeated rank-1 updates at each iteration. Two
well-known update formulae are called Davidon–Fletcher–Powell (DFP) and
Broyden–Fletcher–Goldfarb–Shanno (BFGS). Loss of the positive-definite
condition through round-off error is avoided if rather than updating an
approximation to the inverse of the Hessian, one updates the Cholesky
decomposition of an approximation of the Hessian matrix itself.[citation
needed]

* https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm


* :google:`Broyden–Fletcher–Goldfarb–Shanno GPU`


SIAM J. Sci. Comput., 16(5), 1190–1208. (19 pages)
A Limited Memory Algorithm for Bound Constrained Optimization

Richard H. Byrd, Peihuang Lu, Jorge Nocedal, and Ciyou Zhu
https://doi.org/10.1137/0916069
* https://epubs.siam.org/doi/10.1137/0916069

Parallel L-BFGS-B Algorithm on GPU

* https://hgpu.org/?p=8206


* https://github.com/jwetzl/CudaLBFGS

* https://github.com/nepluno/lbfgsb-gpu

* http://www.cs.columbia.edu/~fyun/lbfgsb/lbfgsb_tech_report.pdf
* https://github.com/painnick/lbfgsb-on-gpu




* https://www.osti.gov/biblio/991824
* ~/gpufit_refs/LMA_Poisson_382281.pdf 

Efficient Levenberg-Marquardt Minimization of the Maximum Likelihood Estimator for Poisson Deviates
T. A. Laurence and B. Chromy


* http://dynopt.cheme.cmu.edu/content/06606/Parest_notes.pdf

Optimization Algorithms for Parameter Estimation and Data Reconciliation
L. T. Biegler
Chemical Engineering Department Carnegie Mellon University Pittsburgh, PA


Maximum Likelihood Estimation on GPUs: Leveraging Dynamic Parallelism
Levenberg-Marquardt algorithm for maximum-likelihood evaluation

* http://on-demand.gputechconf.com/gtc/2015/posters/GTC_2015_Astronomy___Astrophysics_07_P5327_WEB.pdf


* https://arxiv.org/pdf/1201.5885.pdf

Improvements to the Levenberg-Marquardt algorithm for nonlinear least-squares minimization
Mark K. Transtruma, James P. Sethnaa
aLaboratory of Atomic and Solid State Physics, Cornell University, Ithaca, New York 14853, USA







* https://stackoverflow.com/questions/29196139/cholesky-decomposition-with-cuda


* https://docs.nvidia.com/cuda/cusolver/index.html#ormqr-example1


* https://www.cirrelt.ca/DocumentsTravail/CIRRELT-2014-64.pdf


* https://www.researchgate.net/profile/Sverre_Jarp/publication/48410705_Parallelization_of_maximum_likelihood_fits_with_OpenMP_and_CUDA/links/554883360cf26a7bf4dacc60.pdf?inViewer=true&pdfJsDownload=true&disableCoverPage=true&origin=publication_detail

* https://openlab-mu-internal.web.cern.ch/openlab-mu-internal/03_Documents/4_Presentations/Slides/2010-list/CHEP-Maximum-likelihood-fits-on-GPUs.pdf




* https://hgpu.org/?p=4108
* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3027006/pdf/nihms-244740.pdf

Maximum likelihood event estimation and list-mode image reconstruction on GPU hardware

Luca Caucci, Lars R. Furenlid, Harrison H. Barrett
College of Optical Sciences, University of Arizona, 1630 E. University Blvd., Tucson, Arizona 85721, USA
IEEE Nuclear Science Symposium Conference Record (NSS/MIC), 2009
DOI:10.1109/NSSMIC.2009.5402392


https://hgpu.org/?p=10868
GooFit: A library for massively parallelising maximum-likelihood fits


https://hgpu.org/?p=17724
GooFit 2.0

Henry Schreiner, Christoph Hasse, Bradley Hittle, Himadri Pandey, Michael Sokoloff, Karen Tomko
University of Cincinnati, 2600 Clifton Ave, Cincinnati, OH 45220, USA
arXiv:1710.08826 [cs.MS], (21 Oct 2017)



Hydra
* https://arxiv.org/pdf/1711.05683.pdf
* http://on-demand.gputechconf.com/gtc/2017/presentation/S7340-antonio-augusto-alves-hydra-a-framework-for-data-analysis-in-massively-parallel-platforms.pdf
* https://github.com/MultithreadCorner/Hydra


* https://on-demand-gtc.gputechconf.com/gtcnew/on-demand-gtc.php?searchByKeyword=likelihood&searchItems=&sessionTopic=&sessionEvent=&sessionYear=&sessionFormat=&submit=&select=

* https://hgpu.org/?s=likelihood&paged=3

* http://www.idi.ntnu.no/~elster/master-studs/yngve-sneen-lindal/yngve-sneen-lindal-master.pdf
* ~/gpufit_refs/yngve-sneen-lindal-master.pdf



* https://devblogs.nvidia.com/parallel-direct-solvers-with-cusolver-batched-qr/


EOU
}
gpufit-dir(){ echo $(local-base)/env/fit/Gpufit ; }
gpufit-cd(){  cd $(gpufit-dir); }
gpufit-get(){
   local dir=$(dirname $(gpufit-dir)) &&  mkdir -p $dir && cd $dir

    [ ! -d Gpufit ] && git clone https://github.com/gpufit/Gpufit
}


