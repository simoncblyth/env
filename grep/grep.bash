grep-vi(){ vi $BASH_SOURCE ; }
grep-env(){ echo -n ; }
grep-usage(){ cat << EOU
grep notes
============


Find common lines in two files : grep -xF -f a.log b.log 
--------------------------------------------------------------

::

    epsilon:conflict blyth$ grep -xF -f branch.log main.log 
    Examples/Tutorial/python/Tutorial/JUNODetSimModule.py
    Simulation/DetSimV2/PMTSim/src/junoSD_PMT_v2.cc
    Simulation/SimSvc/PMTSimParamSvc/PMTSimParamSvc/PMTSimParamData.h
    cmake/JUNODependencies.cmake
    epsilon:conflict blyth$ 

    -F, --fixed-strings
        Interpret pattern as a set of fixed strings (i.e. force grep to behave as fgrep).

    -x, --line-regexp
        Only input lines selected against an entire fixed string or regular expression are considered to be matching lines.

    -f file, --file=file
         Read one or more newline separated patterns from file.  Empty pattern lines match every input line.  Newlines are not considered part of a
         pattern.  If file is empty, nothing is matched.





EOU
}
