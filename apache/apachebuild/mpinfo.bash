

mpinfo-usage(){
   
   cat << EOU

      mpinfo-location
              apache config snippet for /mpinfo to stdout
              
      mpinfo-conf  
              write the location to $(mpinfo-confpath)
   
   
EOU


}


mpinfo-notes(){

  cat << EON

Internal Server Error

The server encountered an internal error or misconfiguration and was unable to complete your request.
Please contact the server administrator, you@example.com and inform them of the time the error occurred, and anything you might have done that may have caused the error.
More information about this error may be available in the server error log.
Apache/2.0.63 (Unix) DAV/2 mod_python/3.3.1 Python/2.5.1 SVN/1.4.2 Server at cms01.phys.ntu.edu.tw Port 80

EON

}


mpinfo-env(){
   apache-
}

mpinfo-confpath(){
  echo $(apache-confdir)/mpinfo.conf
}

mpinfo-conf(){

   local conf=$(mpinfo-confpath)
   mpinfo-location > $conf
   
   $ASUDO apachectl configtest && $ASUDO apachectl restart 
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



