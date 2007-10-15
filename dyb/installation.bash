
installation-env(){

   export INSTALLATION_FOLDER=$USER_BASE/dayabay/installation
   mkdir -p $INSTALLATION_FOLDER 
   export INSTALLATION_HOME=$INSTALLATION_FOLDER/trunk

}


installation-update(){

    installation-env
 
    cd $INSTALLATION_FOLDER
    test -d trunk || svn --username $USER checkout $DYBSVN/installation/trunk
    cd trunk 
    svn update
      
}
