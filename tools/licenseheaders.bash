# === func-gen- : tools/licenseheaders fgp tools/licenseheaders.bash fgn licenseheaders fgh tools src base/func.bash
licenseheaders-source(){   echo ${BASH_SOURCE} ; }
licenseheaders-edir(){ echo $(dirname $(licenseheaders-source)) ; }
licenseheaders-ecd(){  cd $(licenseheaders-edir); }
licenseheaders-dir(){  echo $LOCAL_BASE/env/tools/licenseheaders ; }
licenseheaders-cd(){   cd $(licenseheaders-dir); }
licenseheaders-vi(){   vi $(licenseheaders-source) $(licenseheaders-dir)/licenseheaders.py ; }
licenseheaders-env(){  elocal- ; }
licenseheaders-usage(){ cat << EOU

Licenseheaders
=================

https://github.com/johann-petrak/licenseheaders

Depends on non-standard regex module 

* https://pypi.org/project/regex/

Install with::

   conda install regex

Python3?

    File "/home/blyth/local/env/tools/licenseheaders/licenseheaders.py", line 373, in read_file
        with open(file, 'r', encoding=args.encoding) as f:
    TypeError: 'encoding' is an invalid keyword argument for this function


EOU
}
licenseheaders-get(){
   local dir=$(dirname $(licenseheaders-dir)) &&  mkdir -p $dir && cd $dir
   git clone git@github.com:simoncblyth/licenseheaders.git
   chmod ugo+x licenseheaders/licenseheaders.py
}

licenseheaders-owner(){    echo Opticks Authors ; }
licenseheaders-projdir(){  echo /tmp/opticks ; }
licenseheaders-projname(){ echo Opticks ; }
licenseheaders-projurl(){  echo https://bitbucket.org/simoncblyth/opticks ; }
licenseheaders-tmpl(){     echo apache-2 ; }
licenseheaders-years(){    echo 2019 ; }

licenseheaders-test-setup()
{
   cd /tmp
   local url=$(licenseheaders-projurl)
   local nam=$(basename $url)
   #[ ! -d $nam ] && hg clone $url
   [ ! -d $nam ] && hg clone ~/opticks
}


licenseheaders--(){
      $(licenseheaders-dir)/licenseheaders.py \
                --additional-extensions c=.hh \
                --dir $(licenseheaders-projdir) \
                --owner="$(licenseheaders-owner)" \
                --projname "$(licenseheaders-projname)" \
                --projurl "$(licenseheaders-projurl)" \
                --tmpl $(licenseheaders-tmpl) \
                --years $(licenseheaders-years) 

}

licenseheaders-info(){ cat << EOI

   licenseheaders-projdir  : $(licenseheaders-projdir)
   licenseheaders-projname : $(licenseheaders-projname)
   licenseheaders-projurl  : $(licenseheaders-projurl)
   licenseheaders-tmpl     : $(licenseheaders-tmpl)
   licenseheaders-years    : $(licenseheaders-years)

EOI
}



