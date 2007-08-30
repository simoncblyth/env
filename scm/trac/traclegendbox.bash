
traclegendbox-get(){

   # http://trac.edgewall.org/wiki/ProcessorBazaar

   local macro=legendbox 
   local name=LegendBox-0.10.py 
   local lame=LegendBox.py
   local url=http://trac.edgewall.org/attachment/wiki/ProcessorBazaar/$name?format=raw
   local dir=$LOCAL_BASE/trac/wiki-macros/$macro
   mkdir -p $dir
   
   cd $dir 
   local cmd="curl -o $lame $url"
   echo $cmd 
   test -f $lame || eval $cmd 

}

traclegendbox-place(){

    local name=${1:-$SCM_TRAC}
    local fold=$SCM_FOLD/tracs/$name
    [ -d "$fold" ] || ( echo  error no folder $fold && exit 1 )

    sudo -u $APACHE2_USER cp $LOCAL_BASE/trac/wiki-macros/legendbox/LegendBox.py $fold/wiki-macros/

}