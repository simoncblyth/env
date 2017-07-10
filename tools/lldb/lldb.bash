# === func-gen- : tools/lldb/lldb fgp tools/lldb/lldb.bash fgn lldb fgh tools/lldb
lldb-src(){      echo tools/lldb/lldb.bash ; }
lldb-source(){   echo ${BASH_SOURCE:-$(env-home)/$(lldb-src)} ; }
lldb-vi(){       vi $(lldb-source) ; }
lldb-env(){      elocal- ; }
lldb-usage(){ cat << EOU

LLDB Experience
=================

Breakpoints
-------------

::

    (lldb) br lis
    (lldb) br dis 1
    1 breakpoints disabled.
    (lldb) br en 1
    1 breakpoints enabled.


Setting Environment
---------------------

::

     (lldb) env DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib


Introspection
---------------

Look at members of a base class::

    (lldb) p m_photons
    (NPY<float> *) $3 = 0x0000000109001840

    (lldb) p *m_photons
    (NPY<float>) $4 = {}

    (lldb) p *(NPYBase*)m_photons
    (NPYBase) $7 = {
      m_dim = 3
      m_ni = 0
      m_nj = 4
      m_nk = 4
      m_nl = 0
      m_sizeoftype = '\x04'
      m_type = FLOAT
      m_buffer_id = -1
      m_buffer_target = -1
      m_aux = 0x0000000000000000
      m_verbose = false
      m_allow_prealloc = false
      m_shape = size=3 {
        [0] = 0
        [1] = 4
        [2] = 4
      }
      m_metadata = "{}"
      m_has_data = true
      m_dynamic = true
    }




EOU
}
lldb-dir(){ echo $(local-base)/env/tools/lldb/tools/lldb-lldb ; }
lldb-cd(){  cd $(lldb-dir); }
lldb-mate(){ mate $(lldb-dir) ; }
lldb-get(){
   local dir=$(dirname $(lldb-dir)) &&  mkdir -p $dir && cd $dir

}
