bitextra-source(){ echo $BASH_SOURCE ; }
bitextra-vi(){     vi $(bitextra-source) ; }
bitextra-usage(){
   package-usage  ${FUNCNAME/-*/}
   cat << EOU
 
    Switch on the extras ... annotation and test summarizer with :
       SUDO=sudo bitextra-prepare



 
    If you find the test result table links from eg 
        http://belle7.nuu.edu.tw/tracs/env/build/demo/18 

    are of the original form :
        http://belle7.nuu.edu.tw/tracs/env/browser/trunk/unittest/demo/package/module_test.py

    rather than :
        http://belle7.nuu.edu.tw/tracs/env/browser/trunk/unittest/demo/package/module_test.py?annotate=bit&rev=1025 

    
    Then you are missing some trac config switching of the TestResultsSummarizer to be smth like :

[components]
bitten.* = enabled
bitten.report.testing.TestResultsSummarizer = disabled
bitextra.* = enabled


    Also see $(env-wikiurl)/Belle7TracFromScratch 


 
EOU

}

bitextra-env(){
  elocal-
  package-
  export BITEXTRA_BRANCH=trunk
}

bitextra-url(){     
   trac-
   echo $(trac-localserver)/repos/tracdev/annobit/$(bitextra-branch) 
}
bitextra-revision(){ echo 111 ; }


bitextra-fix(){
   cd $(bitextra-dir)   
   echo no fix
}


bitextra-package(){   echo bitextra ; }


bitextra-prepare(){

   local name=${FUNCNAME/-*/}
   bitextra-enable $*
  
   ## trac-configure components:$name.TestResultsSummarizer:enabled  is included with the enable


   ## disable the default test summarizer in order for my modified one to take its place    
   
   trac-configure components:bitten.report.testing.TestResultsSummarizer:disabled

   ## verify http://localhost/tracs/workflow/admin/general/plugin
   ## to see that the swap is done OK 
   ##   ... NB you can reconfigure on the fly with trac 0.11 , no need to restart the instance


}





bitextra-branch(){    package-fn  $FUNCNAME $* ; }
bitextra-basename(){  package-fn  $FUNCNAME $* ; }
bitextra-dir(){       package-fn  $FUNCNAME $* ; }  
bitextra-egg(){       package-fn  $FUNCNAME $* ; }
bitextra-get(){       package-fn  $FUNCNAME $* ; }    

bitextra-install(){   package-fn  $FUNCNAME $* ; }
bitextra-uninstall(){ package-fn  $FUNCNAME $* ; } 
bitextra-reinstall(){ package-fn  $FUNCNAME $* ; }
bitextra-enable(){    package-fn  $FUNCNAME $* ; }  

bitextra-status(){    package-fn  $FUNCNAME $* ; } 
bitextra-auto(){      package-fn  $FUNCNAME $* ; } 
bitextra-diff(){      package-fn  $FUNCNAME $* ; } 
bitextra-rev(){       package-fn  $FUNCNAME $* ; } 
bitextra-cd(){        package-fn  $FUNCNAME $* ; } 

bitextra-fullname(){  package-fn  $FUNCNAME $* ; } 
bitextra-update(){    package-fn  $FUNCNAME $* ; } 



