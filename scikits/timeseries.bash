# === func-gen- : scikits/timeseries fgp scikits/timeseries.bash fgn timeseries fgh scikits
timeseries-src(){      echo scikits/timeseries.bash ; }
timeseries-source(){   echo ${BASH_SOURCE:-$(env-home)/$(timeseries-src)} ; }
timeseries-vi(){       vi $(timeseries-source) ; }
timeseries-env(){      elocal- ; }
timeseries-usage(){
  cat << EOU
     timeseries-src : $(timeseries-src)
     timeseries-dir : $(timeseries-dir)


     scikits are unofficial add-ons to scipy and numpy

     scikits.timeseries 
         http://pytseries.sourceforge.net/

     loading timeseries from sqlite DB
         http://projects.scipy.org/scikits/browser/branches/pierregm/hydroclimpy/scikits/hydroclimpy/io/sqlite.py
         http://hydroclimpy.sourceforge.net/introduction.html

     Basis objects have a "freqency" ... is this usable for non regular dates 
         http://mail.scipy.org/pipermail/scipy-user/2008-April/016124.html
             it appears not ... does infilling of missing ticks 
                    


EOU
}
timeseries-dir(){ echo $(local-base)/env/scikits/timeseries ; }
timeseries-cd(){  cd $(timeseries-dir); }
timeseries-mate(){ mate $(timeseries-dir) ; }
timeseries-get(){
   local dir=$(dirname $(timeseries-dir)) &&  mkdir -p $dir && cd $dir
   svn co http://svn.scipy.org/svn/scikits/trunk/timeseries 
}


timeseries-build(){
  timeseries-cd 
  python setup.py install
}
