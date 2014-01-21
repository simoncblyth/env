uncertainties
===============


uncertainties refactor
-----------------------

::

    (chroma_env)delta:chroma blyth$ python -m uncertainties.1to2 -w chroma
    RefactoringTool: Refactored chroma/benchmark.py
    --- chroma/benchmark.py (original)
    +++ chroma/benchmark.py (refactored)
    @@ -45,7 +45,7 @@
                 # first kernel call incurs some driver overhead
                 run_times.append(elapsed)
     
    -    return nphotons/ufloat((np.mean(run_times),np.std(run_times)))
    +    return nphotons/ufloat(np.mean(run_times),np.std(run_times))
     
     def load_photons(number=100, nphotons=500000):
         """Returns the average number of photons moved to the GPU device memory
    @@ -67,7 +67,7 @@
                 # first kernel call incurs some driver overhead
                 run_times.append(elapsed)
     
    -    return nphotons/ufloat((np.mean(run_times),np.std(run_times)))
    +    return nphotons/ufloat(np.mean(run_times),np.std(run_times))
     
     def propagate(gpu_detector, number=10, nphotons=500000, nthreads_per_block=64,
                   max_blocks=1024):
    @@ -95,7 +95,7 @@
                 # first kernel call incurs some driver overhead
                 run_times.append(elapsed)
     
    -    return nphotons/ufloat((np.mean(run_times),np.std(run_times)))
    +    return nphotons/ufloat(np.mean(run_times),np.std(run_times))
     
     @tools.profile_if_possible
     def pdf(gpu_detector, npdfs=10, nevents=100, nreps=16, ndaq=1,
    @@ -153,7 +153,7 @@
                 # first kernel call incurs some driver overhead
                 run_times.append(elapsed)
     
    -    return nevents*nreps*ndaq/ufloat((np.mean(run_times),np.std(run_times)))
    +    return nevents*nreps*ndaq/ufloat(np.mean(run_times),np.std(run_times))
     
     @tools.profile_if_possible
     def pdf_eval(gpu_detector, npdfs=10, nevents=25, nreps=16, ndaq=128,
    @@ -238,7 +238,7 @@
                 # first kernel call incurs some driver overhead
                 run_times.append(elapsed)
     
    -    return nevents*nreps*ndaq/ufloat((np.mean(run_times),np.std(run_times)))
    +    return nevents*nreps*ndaq/ufloat(np.mean(run_times),np.std(run_times))
     
     
     if __name__ == '__main__':
    RefactoringTool: Refactored chroma/likelihood.py
    --- chroma/likelihood.py    (original)
    +++ chroma/likelihood.py    (refactored)
    @@ -101,12 +101,12 @@
             hit_prob = np.maximum(hit_prob, 0.5 / ntotal)
     
             hit_channel_prob = np.log(hit_prob).sum()
    -        log_likelihood = ufloat((hit_channel_prob, 0.0))
    +        log_likelihood = ufloat(hit_channel_prob, 0.0)
     
             # Then include the probability densities of the observed
             # charges and times.
    -        log_likelihood += ufloat((np.log(pdf_prob[self.event.channels.hit]).sum(),
    -                                  0.0))
    +        log_likelihood += ufloat(np.log(pdf_prob[self.event.channels.hit]).sum(),
    +                                  0.0)
             
             return -log_likelihood
     
    @@ -178,7 +178,7 @@
             avg_like = mom1 / mom0
             rms_like = (mom2 / mom0 - avg_like**2)**0.5
             # Don't forget to return a negative log likelihood
    -        return ufloat((-avg_like, rms_like/sqrt(mom0)))
    +        return ufloat(-avg_like, rms_like/sqrt(mom0))
     
     if __name__ == '__main__':
         from chroma.demo import detector as build_detector
    @@ -212,7 +212,7 @@
     
         import matplotlib.pyplot as plt
     
    -    plt.errorbar(x, [v.nominal_value for v in l], [v.std_dev() for v in l])
    +    plt.errorbar(x, [v.nominal_value for v in l], [v.std_dev for v in l])
         plt.xlabel('X Position (m)')
         plt.ylabel('Negative Log Likelihood')
         plt.show()
    RefactoringTool: Refactored chroma/tools.py
    --- chroma/tools.py (original)
    +++ chroma/tools.py (refactored)
    @@ -16,9 +16,9 @@
         return a
     
     def ufloat_to_str(x):
    -    msd = -int(math.floor(math.log10(x.std_dev())))
    +    msd = -int(math.floor(math.log10(x.std_dev)))
         return '%.*f +/- %.*f' % (msd, round(x.nominal_value, msd),
    -                              msd, round(x.std_dev(), msd))
    +                              msd, round(x.std_dev, msd))
     
     def progress(seq):
         "Print progress while iterating over `seq`."
    RefactoringTool: Refactored chroma/histogram/histogram.py
    --- chroma/histogram/histogram.py   (original)
    +++ chroma/histogram/histogram.py   (refactored)
    @@ -82,10 +82,10 @@
                 ma.masked_where(mbins.mask, self.errs[mbins.filled(0)])
     
             if np.iterable(x):
    -            return uarray((value.filled(fill_value), err.filled(fill_err)))
    -        else:
    -            return ufloat((value.filled(fill_value).item(), \
    -                               err.filled(fill_err).item()))
    +            return uarray(value.filled(fill_value), err.filled(fill_err))
    +        else:
    +            return ufloat(value.filled(fill_value).item(), \
    +                               err.filled(fill_err).item())
     
         def interp(self, x):
             """
    @@ -127,9 +127,9 @@
             See sum().
             """
             if width:
    -            return np.dot(np.diff(self.bins), uarray((self.hist, self.errs)))
    -        else:
    -            return np.sum(uarray((self.hist, self.errs)))
    +            return np.dot(np.diff(self.bins), uarray(self.hist, self.errs))
    +        else:
    +            return np.sum(uarray(self.hist, self.errs))
     
         def integrate(self, x1, x2, width=False):
             """
    @@ -157,9 +157,9 @@
     
             if width:
                 return np.dot(np.diff(self.bins[i1:i2+2]),
    -                          uarray((self.hist[i1:i2+1], self.errs[i1:i2+1])))
    -        else:
    -            return np.sum(uarray((self.hist[i1:i2+1], self.errs[i1:i2+1])))
    +                          uarray(self.hist[i1:i2+1], self.errs[i1:i2+1]))
    +        else:
    +            return np.sum(uarray(self.hist[i1:i2+1], self.errs[i1:i2+1]))
     
         def scale(self, c):
             """Scale bin contents and errors by `c`."""
    RefactoringTool: Refactored chroma/histogram/histogramdd.py
    --- chroma/histogram/histogramdd.py (original)
    +++ chroma/histogram/histogramdd.py (refactored)
    @@ -158,7 +158,7 @@
             value, err = ma.array(self.hist[filledbins], mask=valuemask), \
                 ma.array(self.errs[filledbins], mask=valuemask)
     
    -        return uarray((value.filled(fill_value), err.filled(fill_err)))
    +        return uarray(value.filled(fill_value), err.filled(fill_err))
     
         def reset(self):
             """Reset all bin contents/errors to zero."""
    @@ -173,7 +173,7 @@
     
         def usum(self):
             """Return the sum of the bin contents and uncertainty."""
    -        return np.sum(uarray((self.hist, self.errs)))
    +        return np.sum(uarray(self.hist, self.errs))
     
         def scale(self, c):
             """Scale bin values and errors by `c`."""
    RefactoringTool: Files that were modified:
    RefactoringTool: chroma/benchmark.py
    RefactoringTool: chroma/likelihood.py
    RefactoringTool: chroma/tools.py
    RefactoringTool: chroma/histogram/histogram.py
    RefactoringTool: chroma/histogram/histogramdd.py
    (chroma_env)delta:chroma blyth$ 

