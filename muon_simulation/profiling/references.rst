Prior Work
================

* http://dayabay.ihep.ac.cn/tracs/dybsvn/search?q=CPU+profile
* http://dayabay.ihep.ac.cn/tracs/dybsvn/ticket/422 popen trips gperftools, so workaround by providing a file of muons (fopen being OK)
* https://wiki.bnl.gov/dayabay/index.php?title=Profile_r17658
* https://wiki.bnl.gov/dayabay/index.php?title=Optimization

Brett commandline from ticket::

    CPUPROFILE=nuwa.perf LD_PRELOAD=/usr/lib/libprofiler.so.0 $(which python) $(which nuwa.py) -n 25 -o muons10000.root -m 'MDC09b.runMuonRpc.FullChain -T Detector -s 10000'


