# === func-gen- : numerics/pandas/pandas fgp numerics/pandas/pandas.bash fgn pandas fgh numerics/pandas src base/func.bash
pandas-source(){   echo ${BASH_SOURCE} ; }
pandas-edir(){ echo $(dirname $(pandas-source)) ; }
pandas-ecd(){  cd $(pandas-edir); }
pandas-dir(){  echo $LOCAL_BASE/env/numerics/pandas/pandas ; }
pandas-cd(){   cd $(pandas-dir); }
pandas-vi(){   vi $(pandas-source) ; }
pandas-env(){  elocal- ; }
pandas-usage(){ cat << EOU

Pandas 
========


Data formats for Dataframes
-----------------------------

* https://pandas.pydata.org/pandas-docs/stable/io.html#performance-considerations

  Feather is fastest by far 

  * https://pandas.pydata.org/pandas-docs/stable/io.html#feather
  * https://github.com/wesm/feather 
  * https://arrow.apache.org
   

* https://alysivji.github.io/importing-mongo-documents-into-pandas-dataframes.html
* https://en.wikipedia.org/wiki/BSON
* https://github.com/mongodb/bson-numpy/



Giving pandas ROOT to chew on: experiences with the XENON1T Dark Matter experiment 

* http://iopscience.iop.org/article/10.1088/1742-6596/898/4/042003/pdf


EOU
}
pandas-get(){
   local dir=$(dirname $(pandas-dir)) &&  mkdir -p $dir && cd $dir

}
