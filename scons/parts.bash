# === func-gen- : scons/parts fgp scons/parts.bash fgn parts fgh scons
parts-src(){      echo scons/parts.bash ; }
parts-source(){   echo ${BASH_SOURCE:-$(env-home)/$(parts-src)} ; }
parts-vi(){       vi $(parts-source) ; }
parts-env(){      elocal- ; }
parts-usage(){
  cat << EOU
     parts-src : $(parts-src)
     parts-dir : $(parts-dir)

    Open Sourced Extensions to SCons from Intel
       http://parts.tigris.org
       http://parts.tigris.org/doc/GettingStartedGuide.pdf
       http://parts.tigris.org/doc/PartsUserGuide.pdf

    Kludges to get working on PPC ..

      $(python-site)/parts/common.py
           adding to g_arch_map   'ppc':'ppc'  
          
      $(python-site)/parts/plat_info.py 

            val = platform.machine()
            if val == 'Power Macintosh':val = 'ppc'
            return MapArchitecture(val)

     Trying "scons" in  sample/hello sample find that have to specify a target "scons hello"  
     unlike with standard SCons .. which builds everything by default ??
     NB (scons --verbose=all is very verbose)

EOU
}

parts-url(){ echo http://parts.tigris.org/svn/parts/trunk ; }
parts-dir(){ echo $(local-base)/env/scons/parts/parts ; }
parts-cd(){  cd $(parts-dir) ; }
parts-mate(){ mate $(parts-dir) ; }
parts-get(){
   local dir=$(dirname $(dirname $(parts-dir))) &&  mkdir -p $dir && cd $dir
   [ ! -d parts ] && svn co $(parts-url) parts --username guest --password ""
}

parts-install(){
   local msg="=== $FUNCNAME :"
   parts-cd 
   local iwd=$PWD
   #sudo python setup.py install
   python-
   cd $(python-site) 
   local cmd="sudo ln -sf $(parts-dir)/parts parts"
   echo $msg $cmd
   eval $cmd
   cd $iwd
}

