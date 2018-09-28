# === func-gen- : goofit/goofit fgp goofit/goofit.bash fgn goofit fgh goofit
goofit-src(){      echo goofit/goofit.bash ; }
goofit-source(){   echo ${BASH_SOURCE:-$(env-home)/$(goofit-src)} ; }
goofit-vi(){       vi $(goofit-source) ; }
goofit-env(){      elocal- ; }
goofit-usage(){ cat << EOU


* https://goofit.github.io
* https://github.com/GooFit/GooFit
* https://github.com/GooFit
* https://github.com/GooFit/GooTorial



Externals
-----------


Minuit2
~~~~~~~~~

* https://github.com/GooFit/Minuit2
* https://root.cern.ch/root/htmldoc/guides/users-guide/ROOTUsersGuide.html#minuit2-package
* https://github.com/GooFit/Minuit2/blob/master/DEVELOP.md
* http://seal.web.cern.ch/seal/snapshot/work-packages/mathlibs/minuit/doc/doc.html
* http://seal.web.cern.ch/seal/documents/minuit/mntutorial.pdf

Eigen
~~~~~~

* https://github.com/eigenteam/eigen-git-mirror
* https://bitbucket.org/eigen/eigen/src/default/

generics
~~~~~~~~~~

* https://github.com/BryanCatanzaro/generics

NVIDIA GPUs of CUDA compute capability 3.5 and greater, such as the Tesla K20,
support __ldg(), an intrinsic that loads through the read-only texture cache,
and can improve performance in some circumstances. This library allows __ldg to
work on arbitrary types, as detailed below. It also generalizes __shfl() to
shuffle arbitrary types.


modern_cmake
~~~~~~~~~~~~~~

* https://github.com/CLIUtils/modern_cmake

CLI11
~~~~~~~~

* https://github.com/CLIUtils/CLI11

powerful command line parser, with a beautiful, minimal syntax and no
dependencies beyond C++11. It is header only, and comes in a single file form
for easy inclusion in projects. It is easy to use for small projects, but
powerful enough for complex command line projects, and can be customized for
frameworks.



Thrust Usage in Goofit
--------------------------

::

    epsilon:GooFit blyth$ find src -type f -exec grep -H thrust::transform_reduce {} \;

    src/PDFs/physics/Amp4Body.cu:        sumIntegral += thrust::transform_reduce(
    src/PDFs/physics/Amp3Body_IS.cu:            integrals[i] = thrust::transform_reduce(
    src/PDFs/physics/Amp3Body.cu:            (*(integrals[i][j])) = thrust::transform_reduce(
    src/PDFs/physics/Amp3Body.cu:            buffer += thrust::transform_reduce(
    src/PDFs/physics/Amp4Body_TD.cu:        sumIntegral = thrust::transform_reduce(
    src/PDFs/physics/Amp3Body_TD.cu:            (*(integrals[i][j])) = thrust::transform_reduce(
    src/PDFs/GooPdf.cu:    ret = thrust::transform_reduce(
    src/PDFs/GooPdf.cu:    sum = thrust::transform_reduce(

::

    epsilon:GooFit blyth$ find . -name GooPdf.*
    ./python/PDFs/GooPdf.cpp
    ./include/goofit/PDFs/GooPdf.h
    ./src/PDFs/GooPdf.cpp
    ./src/PDFs/GooPdf.cu
    epsilon:GooFit blyth$ 


::

     45 __host__ double GooPdf::reduce_with_metric() const {
     46     double ret;
     47 
     48     double start = 0.0;
     49 
     50     thrust::constant_iterator<int> eventSize(get_event_size());
     51     thrust::constant_iterator<fptype *> arrayAddress(dev_event_array);
     52     thrust::counting_iterator<int> eventIndex(0);
     53 
     54 #ifdef GOOFIT_MPI
     55     size_t entries_to_process = m_iEventsPerTask;
     56 #else
     57     size_t entries_to_process = numEntries;
     58 #endif
     59 
     60     // Calls and sums in parallel:
     61     // logger(0, arrayAddress, eventSize) +
     62     // logger(1, arrayAddress, eventSize) +
     63     // ...
     64 
     65     ret = thrust::transform_reduce(
     66         thrust::make_zip_iterator(thrust::make_tuple(eventIndex, arrayAddress, eventSize)),
     67         thrust::make_zip_iterator(thrust::make_tuple(eventIndex + entries_to_process, arrayAddress, eventSize)),
     68         *logger,
     69         start,
     70         thrust::plus<double>());
     71 
     72 #ifdef GOOFIT_MPI
     73     double r = ret;
     74     MPI_Allreduce(&r, &ret, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
     75 #endif
     76 
     77     return ret;
     78 }


::

     17 __device__ fptype MetricTaker::operator()(thrust::tuple<int, fptype *, int> t) const {
     18     ParameterContainer pc;
     19 
     20     // Calculate event offset for this thread.
     21     int eventIndex       = thrust::get<0>(t);
     22     int eventSize        = thrust::get<2>(t);
     23     fptype *eventAddress = thrust::get<1>(t) + (eventIndex * abs(eventSize));
     24 
     25     fptype events[10];
     26 
     27     // pack our events into the event shared memory.
     28     for(int i = 0; i < abs(eventSize); i++)
     29         events[i] = eventAddress[i];
     30 
     31     int idx = abs(eventSize - 2);
     32     if(idx < 0)
     33         idx = 0;
     34     // fptype obs  = events[idx];
     35     fptype norm = pc.getNormalization(0);
     36 
     37     // Causes stack size to be statically undeterminable.
     38     fptype ret = callFunction(events, pc);
     39 
     40     // Notice assumption here! For unbinned fits the 'eventAddress' pointer won't be used
     41     // in the metric, so it doesn't matter what it is. For binned fits it is assumed that
     42     // the structure of the event is (obs1 obs2... binentry binvolume), so that the array
     43     // passed to the metric consists of (binentry binvolume).
     44 
     45     ret = (*(reinterpret_cast<device_metric_ptr>(d_function_table[pc.funcIdx])))(
     46         ret, eventAddress + (abs(eventSize) - 2), norm);
     47     return ret;
     48 }


src/PDFs/GooPdf.cu::

    370 __device__ fptype callFunction(fptype *eventAddress, ParameterContainer &pc) {
    371     return (*(reinterpret_cast<device_function_ptr>(d_function_table[pc.funcIdx])))(eventAddress, pc);
    372 }


::

    epsilon:GooFit blyth$ find . -name ParameterContainer.*
    ./include/goofit/PDFs/ParameterContainer.h
    ./src/PDFs/ParameterContainer.cu



ParameterContainer latches onto the globals eg d_parameters::

      8 /// This is a container that is used to communicate to the device PDF functions
      9 struct ParameterContainer {
     10     __device__ ParameterContainer();
     11     __device__ ParameterContainer(const ParameterContainer &pc);
     12 
     13     fptype *parameters;
     14     fptype *constants;
     15     fptype *observables;
     16     fptype *normalizations;
     17 
     18     int parameterIdx{0};
     19     int constantIdx{0};
     20     int observableIdx{0};
     21     int normalIdx{0};
     22 
     23     int funcIdx{0};
     24 
     25     inline __device__ fptype getParameter(const int i) { return RO_CACHE(parameters[parameterIdx + i + 1]); }
     26 
     27     inline __device__ fptype getConstant(const int i) { return RO_CACHE(constants[constantIdx + i + 1]); }
     28 
     29     inline __device__ fptype getObservable(const int i) { return RO_CACHE(observables[observableIdx + i + 1]); }
     ////// SCB : mystified, seems that getObservable should return an int ????

     30 
     31     inline __device__ fptype getNormalization(const int i) { return RO_CACHE(normalizations[normalIdx + i + 1]); }
     32 
     33     inline __device__ int getNumParameters() { return (int)RO_CACHE(parameters[parameterIdx]); }
     34 
     35     inline __device__ int getNumConstants() { return (int)RO_CACHE(constants[constantIdx]); }
     36 
     37     inline __device__ int getNumObservables() { return (int)RO_CACHE(observables[observableIdx]); }
     38 
     39     inline __device__ int getNumNormalizations() { return (int)RO_CACHE(normalizations[normalIdx]); }
     40 
     41     // each PDF needs to supply the amount of each array used.
     42     // This function automatically adds +1 for the size.
     43     __device__ void incrementIndex(const int funcs, const int params, const int cons, const int obs, const int norms);
     44 
     45     // slow version, avoid at all costs!
     46     __device__ void incrementIndex();
     47 };
     48 
     49 } // namespace GooFit






::

     07 namespace GooFit {
      8 
      9 // Notice that operators are distinguished by the order of the operands,
     10 // and not otherwise! It's up to the user to make his tuples correctly.
     11 class MetricTaker : public thrust::unary_function<thrust::tuple<int, fptype *, int>, fptype> {
     12   public:
     13     MetricTaker(PdfBase *dat, void *dev_functionPtr);
     14     MetricTaker(int fIdx, int pIdx);
     15 
     16     /// Main operator: Calls the PDF to get a predicted value, then the metric
     17     /// to get the goodness-of-prediction number which is returned to MINUIT.
     18     ///
     19     /// Event number, dev_event_array (pass this way for nvcc reasons), event size
     20     __device__ fptype operator()(thrust::tuple<int, fptype *, int> t) const;
     21 
     22     /// Operator for binned evaluation, no metric.
     23     /// Used in normalization.
     24     ///
     25     /// Event number, event size, normalization ranges (for binned stuff, eg integration)
     26     __device__ fptype operator()(thrust::tuple<int, int, fptype *> t) const;
     27 
     28     /// Update which index we need to use:
     29     __host__ void setFunctionIndex(const int &id) { functionIdx = id; }
     30 
     31   private:
     32     /// Function-pointer index of processing function, eg logarithm, chi-square, other metric.
     33     unsigned int metricIndex;
     34 
     35     /// Function-pointer index of actual PDF
     36     unsigned int functionIdx;
     37 
     38     unsigned int parameters;
     39 };
     40 
     41 } // namespace GooFit



::

    epsilon:GooFit blyth$ find . -name GooPdf.* -exec grep -H logger {} \;
    ./include/goofit/PDFs/GooPdf.h:    std::shared_ptr<MetricTaker> logger;
    ./src/PDFs/GooPdf.cpp:    logger = std::make_shared<MetricTaker>(this, getMetricPointer(fitControl->getMetric()));
    ./src/PDFs/GooPdf.cu:    // logger(0, arrayAddress, eventSize) +
    ./src/PDFs/GooPdf.cu:    // logger(1, arrayAddress, eventSize) +
    ./src/PDFs/GooPdf.cu:        *logger,
    ./src/PDFs/GooPdf.cu:    logger->setFunctionIndex(functionIdx);
    ./src/PDFs/GooPdf.cu:    // logger(0, eventSize, arrayAddress) +
    ./src/PDFs/GooPdf.cu:    // logger(1, eventSize, arrayAddress) +
    ./src/PDFs/GooPdf.cu:        *logger,
    ./src/PDFs/GooPdf.cu:    // logger(0, arrayAddress, eventSize)
    ./src/PDFs/GooPdf.cu:    // logger(1, arrayAddress, eventSize)
    ./src/PDFs/GooPdf.cu:        *logger);
    ./src/PDFs/GooPdf.cu:    // Ensure that we properly populate *logger with the correct metric
    epsilon:GooFit blyth$ 

::

    epsilon:GooFit blyth$ find . -type f -exec grep -H dev_event_array {} \;
    ./include/goofit/PDFs/detail/Globals.h:extern fptype *dev_event_array;
    ./include/goofit/PDFs/MetricTaker.h:    /// Event number, dev_event_array (pass this way for nvcc reasons), event size
    ./docs/documentation.md:gooMalloc((void**) &dev_event_array, dimensions*numEntries*sizeof(fptype));
    ./docs/documentation.md:cudaMemcpy(dev_event_array, host_array, dimensions*numEntries*sizeof(fptype), cudaMemcpyHostToDevice);
    ./docs/documentation.md:uses the `dev_event_array` either as a list of events or a list of bins.
    ./src/PDFs/detail/Globals.cpp:fptype *dev_event_array;
    ./src/PDFs/physics/Amp4Body.cu:    thrust::constant_iterator<fptype *> dataArray(dev_event_array);
    ./src/PDFs/physics/Amp4Body.cu:    dev_event_array = thrust::raw_pointer_cast(DS->data());
    ./src/PDFs/physics/Amp3Body_IS.cu:    thrust::constant_iterator<fptype *> dataArray(dev_event_array);
    ./src/PDFs/physics/Amp3Body.cu:    thrust::constant_iterator<fptype *> dataArray(dev_event_array);
    ./src/PDFs/physics/Amp4Body_TD.cu:    thrust::constant_iterator<fptype *> dataArray(dev_event_array);
    ./src/PDFs/physics/Amp4Body_TD.cu:    dev_event_array = thrust::raw_pointer_cast(DS->data());
    ./src/PDFs/physics/Amp4Body_TD.cu:    thrust::constant_iterator<fptype *> arrayAddress(dev_event_array);
    ./src/PDFs/physics/Amp4Body_TD.cu:    gooFree(dev_event_array);
    ./src/PDFs/physics/Amp3Body_TD.cu:    thrust::constant_iterator<fptype *> dataArray(dev_event_array);
    ./src/PDFs/GooPdf.cu:    thrust::constant_iterator<fptype *> arrayAddress(dev_event_array);
    ./src/PDFs/GooPdf.cu:    thrust::constant_iterator<fptype *> arrayAddress(dev_event_array);
    ./src/PDFs/PdfBase.cu:    if(dev_event_array) {
    ./src/PDFs/PdfBase.cu:        gooFree(dev_event_array);
    ./src/PDFs/PdfBase.cu:        dev_event_array  = nullptr;
    ./src/PDFs/PdfBase.cu:        gooMalloc((void **)&dev_event_array, dimensions * mycount * sizeof(fptype));
    ./src/PDFs/PdfBase.cu:        MEMCPY(dev_event_array,
    ./src/PDFs/PdfBase.cu:        gooMalloc((void **)&dev_event_array, dimensions * numEntries * sizeof(fptype));
    ./src/PDFs/PdfBase.cu:        MEMCPY(dev_event_array, host_array, dimensions * numEntries * sizeof(fptype), cudaMemcpyHostToDevice);
    ./src/PDFs/PdfBase.cu:        gooMalloc((void **)&dev_event_array, dimensions * mycount * sizeof(fptype));
    ./src/PDFs/PdfBase.cu:        MEMCPY(dev_event_array,
    ./src/PDFs/PdfBase.cu:        gooMalloc((void **)&dev_event_array, dimensions * numEntries * sizeof(fptype));
    ./src/PDFs/PdfBase.cu:        MEMCPY(dev_event_array, host_array, dimensions * numEntries * sizeof(fptype), cudaMemcpyHostToDevice);
    epsilon:GooFit blyth$ 




EOU
}
goofit-dir(){ echo $(local-base)/env/goofit/GooFit ; }
goofit-cd(){  cd $(goofit-dir); }
goofit-get(){
   local dir=$(dirname $(goofit-dir)) &&  mkdir -p $dir && cd $dir

   #[ ! -d GooFit ] && git clone git@github.com:simoncblyth/GooFit.git  
   [ ! -d GooFit ] && git clone git://github.com/GooFit/GooFit.git --recursive
   

}
