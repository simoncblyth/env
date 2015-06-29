# === func-gen- : base/cpp.bash fgp base/cpp.bash fgn cpp
cpp-src(){      echo base/cpp.bash ; }
cpp-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cpp-src)} ; }
cpp-vi(){       vi $(cpp-source) ; }
cpp-env(){      elocal- ; }
cpp-usage(){
  cat << EOU

C++
====


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



