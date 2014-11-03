# === func-gen- : numpy/rlibnpy fgp numpy/rlibnpy.bash fgn rlibnpy fgh numpy
rlibnpy-src(){      echo numpy/rlibnpy/rlibnpy.bash ; }
rlibnpy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rlibnpy-src)} ; }
rlibnpy-vi(){       vi $(rlibnpy-source) ; }
rlibnpy-env(){      elocal- ; }
rlibnpy-usage(){ cat << EOU

Reimplementation of libnpy as single C++ header
=================================================

Original source. 

   git clone https://gist.github.com/5656056.git rlibnpy 

Modified to allow use with memory buffers in addition 
to files.

::

    delta:rlibnpy blyth$ xxd onedim_b_npy 
    0000000: 934e 554d 5059 0100 4600 7b27 6465 7363  .NUMPY..F.{'desc
    0000010: 7227 3a20 273c 6934 272c 2027 666f 7274  r': '<i4', 'fort
    0000020: 7261 6e5f 6f72 6465 7227 3a20 5472 7565  ran_order': True
    0000030: 2c20 2773 6861 7065 273a 2028 362c 292c  , 'shape': (6,),
    0000040: 207d 2020 2020 2020 2020 2020 2020 200a   }             .
    0000050: 0100 0000 0200 0000 0300 0000 0400 0000  ................
    0000060: 0500 0000 0600 0000                      ........



EOU
}
rlibnpy-dir(){ echo $(local-base)/env/numpy/rlibnpy ; }
rlibnpy-sdir(){ echo $(env-home)/numpy/rlibnpy ; }
rlibnpy-tdir(){ echo /tmp/env/numpy/rlibnpy ; }
rlibnpy-cd(){  cd $(rlibnpy-dir); }
rlibnpy-scd(){  cd $(rlibnpy-sdir); }
rlibnpy-tcd(){  mkdir -p $(rlibnpy-tdir) && cd $(rlibnpy-tdir); }


rlibnpy-test(){
   rlibnpy-tcd
   clang $(rlibnpy-sdir)/numpy_test.cc -lstdc++ -o $FUNCNAME && ./$FUNCNAME &&  $(rlibnpy-sdir)/load.py *_npy  

}


