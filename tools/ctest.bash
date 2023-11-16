ctest-vi(){ vi $BASH_SOURCE ; }
ctest-env(){ echo -n ; }
ctest-usage(){ cat << EOU
ctest
======


tips
------

::

    ctest -N                   ## list tests 
    ctest --output-on-failure  ## run all tests 
    ctest -N --rerun-failed    ## list failed tests without running them
    ctest -I 10,20             ## run range of tests selected by index



test fixtures ? Added in CMake 3.7.0
--------------------------------------

* https://crascit.com/2016/10/18/test-fixtures-with-cmake-ctest/




EOU
}
