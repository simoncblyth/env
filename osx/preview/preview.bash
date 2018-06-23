# === func-gen- : osx/preview/preview fgp osx/preview/preview.bash fgn preview fgh osx/preview
preview-src(){      echo osx/preview/preview.bash ; }
preview-source(){   echo ${BASH_SOURCE:-$(env-home)/$(preview-src)} ; }
preview-vi(){       vi $(preview-source) ; }
preview-env(){      elocal- ; }
preview-usage(){ cat << EOU


Preview.app image editing
===========================

* https://support.apple.com/en-us/HT201740



Printing a PDF that gives errors with a printer
-------------------------------------------------

* File > Export then select PNG as format 
  (it manages to have multiple pages in the PNG) 



Scaledown retina screen capture by factor 2
----------------------------------------------

* hit edit button, select size icon, change dpi from 144 to 72
  it saves automatically 


Combine multiple PDFs into one PDF 
---------------------------------------

* see osx_automator_workflows


Combine multiple PNGs into one PDF
--------------------------------------

* see osx_automator_workflows








EOU
}
preview-dir(){ echo $(local-base)/env/osx/preview/osx/preview-preview ; }
preview-cd(){  cd $(preview-dir); }
preview-mate(){ mate $(preview-dir) ; }
preview-get(){
   local dir=$(dirname $(preview-dir)) &&  mkdir -p $dir && cd $dir

}
