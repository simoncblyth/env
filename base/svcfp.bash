# === func-gen- : base/svcfp fgp base/svcfp.bash fgn svcfp fgh base
svcfp-src(){      echo base/svcfp.bash ; }
svcfp-source(){   echo ${BASH_SOURCE:-$(env-home)/$(svcfp-src)} ; }
svcfp-vi(){       vi $(svcfp-source) ; }
svcfp-env(){      elocal- ; }
svcfp-usage(){
  cat << EOU
     svcfp-src : $(svcfp-src)
     svcfp-dir : $(svcfp-dir)

       Functionality not used in anger ... 

     svcfp-dump
     svcfp-getset

         ConfigParser based dumping and get/get 
         (as ConfigObj does not handle ";" comments)


   Problems with config automation 
     1) ConfigObj doesnt handle ";" comments... and does not preserve spacing of inline # comments 
     2) Supervisor uses ConfigParser internally ... but this drops comments 
  
   Went with ConfigParser in ini-edit re-implementation ~/e/base/ini_cp.py   


EOU
}

svcfp-dump(){ $FUNCNAME- | python ; }
svcfp-dump-(){ cat << EOD
from ConfigParser import ConfigParser
c = ConfigParser()
c.read("$(sv-;sv-confpath)")
for section in c.sections():
    print section
    for option in c.options(section):
        print " ", option, "=", c.get(section, option)
EOD
}

svcfp-getset(){  $FUNCNAME- | python - $* ; }
svcfp-getset-(){ cat << EOD
## not used in anger ... see sv-ini
import sys
from ConfigParser import ConfigParser
c = ConfigParser()
c.read("$(sv-;sv-confpath)")
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



