opticksgeo-src(){      echo opticksgeo/opticksgeo.bash ; }
opticksgeo-source(){   echo ${BASH_SOURCE:-$(env-home)/$(opticksgeo-src)} ; }
opticksgeo-vi(){       vi $(opticksgeo-source) ; }
opticksgeo-usage(){ cat << EOU



EOU
}

opticksgeo-env(){      elocal- ; opticks- ; }

opticksgeo-dir(){  echo $(env-home)/opticksgeo ; }
opticksgeo-sdir(){ echo $(env-home)/opticksgeo ; }
opticksgeo-tdir(){ echo $(env-home)/opticksgeo/tests ; }
opticksgeo-idir(){ echo $(opticks-idir); } 
opticksgeo-bdir(){ echo $(opticks-bdir)/opticksgeo ; }  

opticksgeo-cd(){   cd $(opticksgeo-dir); }
opticksgeo-icd(){  cd $(opticksgeo-idir); }
opticksgeo-bcd(){  cd $(opticksgeo-bdir); }
opticksgeo-scd(){  cd $(opticksgeo-sdir); }
opticksgeo-tcd(){  cd $(opticksgeo-tdir); }

opticksgeo-wipe(){ local bdir=$(opticksgeo-bdir) ; rm -rf $bdir ; }

opticksgeo-name(){ echo OpticksGeometry ; }
opticksgeo-tag(){  echo OKGEO ; }

opticksgeo--(){        opticks--     $(opticksgeo-bdir) ; }
opticksgeo-ctest(){    opticks-ctest $(opticksgeo-bdir) $* ; }
opticksgeo-genproj(){  opticksgeo-scd ; opticks-genproj $(opticksgeo-name) $(opticksgeo-tag) ; }
opticksgeo-gentest(){  opticksgeo-tcd ; opticks-gentest ${1:-OpticksGeometry} $(opticksgeo-tag) ; }

opticksgeo-txt(){   vi $(opticksgeo-sdir)/CMakeLists.txt $(opticksgeo-tdir)/CMakeLists.txt ; }


