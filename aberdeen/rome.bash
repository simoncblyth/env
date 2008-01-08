


rome-env(){

   export ROME_NAME=rome_v2.9
   #export ROME_NAME=trunk
   export ROMESYS=$LOCAL_BASE/rome/$ROME_NAME

   alias romebuilder="$ROMESYS/bin/romebuilder.exe"

}

rome-make(){

  rome-env
  cd $ROMESYS
  make
   
}

rome-get(){
   

   rome-env
   
   local nik=rome
   local name=$ROME_NAME
   local tgz=$name.tar.gz

   local url 

   if [ "$ROME_NAME" == "trunk" ]; then
       url="svn+ssh://svn@savannah.psi.ch/afs/psi.ch/project/meg/svn/rome/trunk/rome"
   else    
       url="http://savannah.psi.ch/viewcvs/tags/$name?root=rome&view=tar"
   fi


   local dir=$LOCAL_BASE/$nik
   $SUDO mkdir -p $dir && $SUDO chown $USER $dir
   cd $dir

   local cmd

   if [ "$ROME_NAME" == "trunk" ]; then
      test -d $ROME_NAME && cmd="svn up $ROME_NAME" || cmd="svn co $url $ROME_NAME"
   
      echo $cmd
      eval $cmd 
   
   
   else
      test -f $tgz || curl -o $tgz $url
      test -d $name || tar zxvf $tgz 
   fi

   

}

