# === func-gen- : proj/key4hep/e4 fgp proj/key4hep/e4.bash fgn e4 fgh proj/key4hep src base/func.bash
e4-source(){   echo ${BASH_SOURCE} ; }
e4-edir(){ echo $(dirname $(e4-source)) ; }
e4-ecd(){  cd $(e4-edir); }
e4-dir(){  echo $LOCAL_BASE/env/proj/key4hep/edm4hep ; }
e4-cd(){   cd $(e4-dir); }
e4-vi(){   vi $(e4-source) ; }
e4-env(){  elocal- ; }
e4-usage(){ cat << EOU


https://indico.jlab.org/event/420/contributions/8308/attachments/6909/9428/210504_sailer_key4hep.pdf

For a high degree of interoperability, EDM4hep provides a
common event data model
Using podio to manage the EDM (described by yaml) and
easily change the persistency layer (ROOT, SIO, . . . )
http://github.com/key4hep/edm4hep
EDM4hep data model based on LCIO and FCC-edm
Key4hep triggered many developments for podio:
I SIO backend
I new templating
I metadata
I multi-threading investigations
I schema evolution
EDM4hep to be used at any stage of Key4he



EOU
}
e4-get(){
   local dir=$(dirname $(e4-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d edm4hep ] && git clone http://github.com/key4hep/edm4hep 
}
