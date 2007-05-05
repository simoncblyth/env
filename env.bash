
ENV_BASE=env

env-i(){ [ -r $HOME/$ENV_BASE/env.bash ] && . $HOME/$ENV_BASE/env.bash ; }

env-find(){
  q=${1:-dummy}
  cd $HOME/$ENV_BASE
  find . -name '*.*' -exec grep -H $q {} \;
}

env-x-pkg(){

  X=${1:-$TARGET_TAG}

  if [ "$X" == "ALL" ]; then
    xs="P H G1"
  else
    xs="$X"
  fi

  for x in $xs
  do	
     base-x-pkg $x
     scm-x-pkg $x
     dyw-x-pkg $x
  done

}


env-x-pkg-not-working(){

  X=${1:-$TARGET_TAG}

  iwd=$(pwd)
  cd $HOME/$ENV_BASE
  dirs=$(ls -1)
  for d in $dirs
  do
    if [ -d $d ]; then
  		cmd="$d-x-pkg $X"
		echo $cmd
		eval $cmd
 	fi 	
  done

}




