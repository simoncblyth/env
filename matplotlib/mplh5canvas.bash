# === func-gen- : matplotlib/mplh5canvas fgp matplotlib/mplh5canvas.bash fgn mplh5canvas fgh matplotlib
mplh5canvas-src(){      echo matplotlib/mplh5canvas.bash ; }
mplh5canvas-source(){   echo ${BASH_SOURCE:-$(env-home)/$(mplh5canvas-src)} ; }
mplh5canvas-vi(){       vi $(mplh5canvas-source) ; }
mplh5canvas-env(){      elocal- ; }
mplh5canvas-usage(){
  cat << EOU
     mplh5canvas-src : $(mplh5canvas-src)
     mplh5canvas-dir : $(mplh5canvas-dir)

         special backend that spawns a server rendition of plot .... 
    
     http://www.google.com.tw/search?q=mplh5canvas 

     http://code.google.com/p/mplh5canvas/wiki/Installation
     http://code.google.com/p/mplh5canvas/issues/list

     http://trac.sagemath.org/sage_trac/ticket/9471

     http://comments.gmane.org/gmane.comp.python.matplotlib.devel/8716
           firefox problematic ... about integration in to web apps 
         
     http://www.mail-archive.com/matplotlib-devel@lists.sourceforge.net/msg06873.html

EOU
}
mplh5canvas-dir(){ echo $(local-base)/env/matplotlib/mplh5canvas ; }
mplh5canvas-cd(){  cd $(mplh5canvas-dir); }
mplh5canvas-mate(){ mate $(mplh5canvas-dir) ; }
mplh5canvas-get(){
   local dir=$(dirname $(mplh5canvas-dir)) &&  mkdir -p $dir && cd $dir
   svn co http://mplh5canvas.googlecode.com/svn/trunk mplh5canvas
}

mplh5canvas-build(){
   mplh5canvas-cd
   python setup.py install
}
