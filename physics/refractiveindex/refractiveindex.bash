# === func-gen- : physics/refractiveindex/refractiveindex fgp physics/refractiveindex/refractiveindex.bash fgn refractiveindex fgh physics/refractiveindex
refractiveindex-src(){      echo physics/refractiveindex/refractiveindex.bash ; }
refractiveindex-source(){   echo ${BASH_SOURCE:-$(env-home)/$(refractiveindex-src)} ; }
refractiveindex-vi(){       vi $(refractiveindex-source) ; }
refractiveindex-env(){      elocal- ; }
refractiveindex-usage(){ cat << EOU


Refractive Indices
===================

::

    a = refractiveindex("tmp/glass/schott/F2.csv")   # from 334. nm
    b = refractiveindex("tmp/main/H2O/Hale.csv")     # from 200. nm

    #
    # does my wavelength domain need to start so low 80nm, thats beyond far UV 
    # it causes problems with artificial plateaus 
    #  
    # probably its following chroma ?
    #


* https://en.wikipedia.org/wiki/Sellmeier_equation


EOU
}

refractiveindex-dir(){ echo $LOCAL_BASE/env/physics/refractiveindex ; }
refractiveindex-edir(){ echo $(env-home)/physics/refractiveindex ; }
refractiveindex-ecd(){  cd $(refractiveindex-edir); }
refractiveindex-cd(){   cd $(refractiveindex-dir); }

refractiveindex-get(){
   $(refractiveindex-edir)/refractiveindex.py 
}

refractiveindex-i(){
   refractiveindex-ecd

   ls -l 
   i

 
}


