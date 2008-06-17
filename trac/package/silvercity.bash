
silvercity-usage(){

   package-fn $FUNCNAME $*

cat << EOU


  another provider of syntax highlighting to trac...
  installed principally to avoid the error messages in the trac.log and
  associated performance hit

      http://trac.edgewall.org/wiki/TracSyntaxColoring
      http://trac.edgewall.org/wiki/SilverCity 
       
       
   they seem to have abandoned this repo ???    
      https://silvercity.svn.sourceforge.net/svnroot/silvercity/ 
    
    
     python setup.py install  
    # Writing /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/SilverCity-0.9.7-py2.5.egg-info
                
       
EOU

}


silvercity-env(){
   elocal-
   package-
   
   export SILVERCITY_BRANCH=SilverCity-0.9.7.tar.gz
}

silvercity-url(){
   echo http://nchc.dl.sourceforge.net/sourceforge/silvercity/$(silvercity-branch)
}



silvercity-pkgname(){
  ## needed when the script name does not correspond to the python package name
   echo SilverCity
}


silvercity-test(){
   python -c "import SilverCity"
}

silvercity-branch(){    package-fn $FUNCNAME $* ; }
silvercity-basename(){  package-fn $FUNCNAME $* ; }
silvercity-dir(){       package-fn $FUNCNAME $* ; } 

silvercity-get(){       package-fn $FUNCNAME $* ; }

silvercity-install(){   package-fn $FUNCNAME $* ; }
silvercity-uninstall(){ package-fn $FUNCNAME $* ; }
silvercity-reinstall(){ package-fn $FUNCNAME $* ; }
silvercity-enable(){    package-fn $FUNCNAME $* ; }

silvercity-status(){    package-fn $FUNCNAME $* ; }
silvercity-auto(){      package-fn $FUNCNAME $* ; }
silvercity-diff(){      package-fn $FUNCNAME $* ; } 
silvercity-rev(){       package-fn $FUNCNAME $* ; } 
silvercity-cd(){        package-fn $FUNCNAME $* ; }

silvercity-fullname(){  package-fn $FUNCNAME $* ; }
silvercity-update(){    package-fn $FUNCNAME $* ; }

