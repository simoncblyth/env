

svntest-usage(){

   cat << EOU

     svntest-repos
     svntest-tracs
     svntest-mpinfo

EOU

}

svntest-env(){
   elocal-
}

svntest-names(){
  echo env dybsvn aberdeen tracdev
}

svntest-repos(){  svntest-pfx repos $* ; }
svntest-tracs(){  svntest-pfx tracs $* ; }

svntest-pfx(){
  local p=${1:-repos}   
  local t=${2:-localhost}
  local url
  for name in $(svntest-names)
  do
     url=http://$t/$pfx/$name/
     curl $url
  done
}


