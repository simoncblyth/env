# === func-gen- : base/cpp.bash fgp base/cpp.bash fgn cpp
cpp-src(){      echo base/cpp.bash ; }
cpp-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cpp-src)} ; }
cpp-vi(){       vi $(cpp-source) ; }
cpp-env(){      elocal- ; }
cpp-usage(){
  cat << EOU

C++
====


Forward Declare Template Class
-------------------------------

::

    template <typename T> class Constructor ;


Explicit instanciation of template class
------------------------------------------

::

    template class Constructor<OctreeNode> ;


Forward Instanciation of template method
-----------------------------------------

::

    template void TBuf::upload<float>(NPY<float>*) const ;


Instanciation visible across libs::

    270 template class THRAP_API TSparse<unsigned long long> ;
    271 template class THRAP_API TSparse<int> ;
    272 
    273 template THRAP_API void TSparse<unsigned long long>::apply_lookup<unsigned char>(CBufSlice target);
    274 template THRAP_API void TSparse<int>::apply_lookup<char>(CBufSlice target);



C++11
-------

* http://herbsutter.com/elements-of-modern-c-style/

Versions
---------


     cpp-src : $(cpp-src)

   hfag    "3.2.3 20030502 (Red Hat Linux 3.2.3-59)"
   grid1   "3.2.3 20030502 (Red Hat Linux 3.2.3-59)" 

   cms02   "3.4.6 20060404 (Red Hat 3.4.6-10)"
   cms01   "3.4.6 20060404 (Red Hat 3.4.6-11)"
   belle7  "4.1.2 20070626 (Red Hat 4.1.2-14)"

EOU
}
cpp-version(){ cpp-defines | grep __VERSION__ ; }
cpp-defines(){ echo | /usr/bin/c++ -x c++ -E -dD - ; }



