# === func-gen- : nuwa/detdesc/detdesc fgp nuwa/detdesc/detdesc.bash fgn detdesc fgh nuwa/detdesc
detdesc-src(){      echo nuwa/detdesc/detdesc.bash ; }
detdesc-source(){   echo ${BASH_SOURCE:-$(env-home)/$(detdesc-src)} ; }
detdesc-vi(){       vi $(detdesc-source) ; }
detdesc-env(){      elocal- ; }
detdesc-usage(){ cat << EOU

DetDesc
=======


Reminders
-----------

* lots of geometry is generated by python scripts, 
  so to follow along need to examine InstallArea not just sources 

::

    ssh G5 
    cd /home/blyth/local/env/dyb/NuWa-trunk/dybgaudi

    [blyth@ntugrid5 dybgaudi]$ find . -type d -name AD
    ./InstallArea/python/XmlDetDescGen/AD
    ./InstallArea/python/MiniDryRunXmlDetDescGen/AD

    cd /home/blyth/local/env/dyb/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/AD





EOU
}
detdesc-dir(){ echo $(local-base)/env/nuwa/detdesc/nuwa/detdesc-detdesc ; }
detdesc-cd(){  cd $(detdesc-dir); }
detdesc-mate(){ mate $(detdesc-dir) ; }
detdesc-get(){
   local dir=$(dirname $(detdesc-dir)) &&  mkdir -p $dir && cd $dir

}
