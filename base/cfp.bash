# === func-gen- : base/cfp fgp base/cfp.bash fgn cfp fgh base
cfp-src(){      echo base/cfp.bash ; }
cfp-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cfp-src)} ; }
cfp-vi(){       vi $(cfp-source) ; }
cfp-env(){      elocal- ; }
cfp-usage(){
  cat << EOU
     cfp-src : $(cfp-src)
     cfp-dir : $(cfp-dir)

     Namechange from svcfp- to cfp- to reflect the generality of 
     ini editing ... 

     svcfp-  functionality has not used in anger ... 

     cfp-dump
     cfp-getset
         ConfigParser based dumping and get/get 
         (as ConfigObj does not handle ";" comments)



     cfp-multifile-tst
         missing files silently ignored ...
         last file wins 




   Problems with config automation 
     1) ConfigObj doesnt handle ";" comments... and does not preserve spacing of inline # comments 
     2) Supervisor uses ConfigParser internally ... but this drops comments 
  
   Went with ConfigParser in ini-edit re-implementation ~/e/base/ini_cp.py   

EOU
}

cfp-defpath(){ sv-;sv-confpath ; }
cfp-path(){ echo ${CFP_PATH:-$(cfp-defpath)} ; }

cfp-dump(){ $FUNCNAME- | python ; }
cfp-dump-(){ cat << EOD
from ConfigParser import ConfigParser
c = ConfigParser()
c.read("$(cfp-path)")
for section in c.sections():
    print section
    for option in c.options(section):
        print " ", option, "=", c.get(section, option)
EOD
}

cfp-getset(){  $FUNCNAME- | python - $* ; }
cfp-getset-(){ cat << EOD
import sys
from ConfigParser import ConfigParser
c = ConfigParser()

path = "$(cfp-path)"
c.read(path.split(":"))
argv = sys.argv[1:]

if len(argv) == 0:
   c.write(sys.stdout)
elif len(argv) == 1:
   section = argv[0]
   for option in c.options(section):
       print " ", option, "=", c.get(section, option)
elif len(argv) == 2:
   print c.get(*argv)
elif len(argv) == 3:
   c.set(*argv)
   print "; $FUNCNAME set %s " %  repr(argv)
   c.write(sys.stdout)  
else:
   pass

EOD
}



cfp-multifile-tst(){

  local iwd=$PWD
  local tmp=/tmp/$USER/env/$FUNCNAME && mkdir -p $tmp
  cd $tmp

  printf "[aaa]\nared=1\nablue=1\n[bbb]\nbred=1\n" > a.cnf
  printf "[aaa]\nared=2\nablue=2\n[bbb]\nbred=2\n[ccc]\ncred=3\n" > b.cnf

  [ "$(CFP_PATH=a.cnf:b.cnf cfp-getset aaa ared)" == "2" ] && echo last file : b.cnf wins  
  [ "$(CFP_PATH=b.cnf:a.cnf cfp-getset aaa ared)" == "1" ] && echo last file : a.cnf wins 
  [ "$(CFP_PATH=a.cnf:b.cnf:c.cnf cfp-getset ccc cred)" == "3" ] && echo missing files are ignored  

  cd $iwd
}
