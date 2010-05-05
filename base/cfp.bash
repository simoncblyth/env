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
c.read("$(cfp-path)")
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



