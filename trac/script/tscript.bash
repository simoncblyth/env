
tscript-env(){
  echo -n
}

tscript-usage(){
  cat << EOU

    tscript-t2l path/to/wikitext/file
        convert to latex, putting the 

    echo "== Hello ==" | tscript-t2l

EOU

}

tscript-t2l-py(){ echo $ENV_HOME/trac/script/t2l.py ; }

tscript-t2l(){
   case "$#" in
      0) cat - | python `tscript-t2l-py` ;;
      *) python `tscript-t2l-py` $* ;;
   esac  

}

tscript-t2l-test(){
  tscript-t2l /Users/blyth/workflow/admin/reports/nuu-report-oct-2008.txt
}

