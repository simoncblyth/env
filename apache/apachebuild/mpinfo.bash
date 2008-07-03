

mpinfo-usage(){
   
   cat << EOU

      mpinfo-conf :
            
   
   
EOU


}


mpinfo-env(){
   apache-
}

mpinfo-conf(){

   local conf=$(apache-confdir)/mpinfo.conf
   mpinfo-location > $conf
   
   sudo apachectl configtest && sudo apachectl restart 
   echo open http://$(apache-target)/mpinfo
}


mpinfo-location(){
cat << EOL
<Location /mpinfo>
   SetHandler mod_python
   PythonHandler mod_python.testhandler
</Location>
EOL

}



