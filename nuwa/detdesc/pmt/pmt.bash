# === func-gen- : nuwa/detdesc/pmt/pmt fgp nuwa/detdesc/pmt/pmt.bash fgn pmt fgh nuwa/detdesc/pmt
pmt-src(){      echo nuwa/detdesc/pmt/pmt.bash ; }
pmt-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pmt-src)} ; }
pmt-vi(){       vi $(pmt-source) ; }
pmt-env(){      elocal- ; }
pmt-usage(){ cat << EOU





EOU
}
pmt-dir(){ echo $(local-base)/env/dyb/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/PMT ; }
pmt-edir(){ echo $(env-home)/nuwa/detdesc/pmt ; }
pmt-export(){  
    export PMT_DIR=$(pmt-dir) 
}

pmt-cd(){  cd $(pmt-dir); }
pmt-ecd(){ cd $(pmt-edir) ; }

pmt-run(){ 
   pmt-export
   python $(pmt-edir)/pmt.py 
}






