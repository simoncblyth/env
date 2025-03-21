mail-vi(){   vi $BASH_SOURCE ; }
mail-env(){
  elocal-
}

mail-usage(){

  cat << EOU

Debugging Usage of Mail.app 
==============================

Issues
--------

Many tens of message copies appearing in Mail.app Trash
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This makes it difficult to search as getting hundreds
of hits from the duplicitous Trash.
Possibly a gmail/Mail.app incompatibility 
regards autosave.

* :google:`OSX Mail.app Trash message hundreds`

* http://apple.stackexchange.com/questions/36894/apple-mail-app-search-filter-brings-the-same-email-hundred-times
* https://discussions.apple.com/thread/5468664?start=15&tstart=0


FUNCTIONS
-----------

mail-run 
       exit Mail.app then run this to start it up with a live view of error messages 


EOU

}

mail-db-path(){      echo /tmp/$USER/env/mail.db ; }
mail-db-origpath(){  echo ~/Library/Mail/Envelope\ Index ; }
mail-db-(){
   local src="$(mail-db-origpath)"
   local tgt="$(mail-db-path)"
   mkdir -p $(dirname $tgt)
   [ "$tgt" -nt "$src" -o ! -f "$tgt" ] && echo $msg copying from "$src" to "$tgt" && cp "$src" "$tgt"
   sqlite3 $(mail-db-path)
}

mail-offline-cache(){ echo ~/Library/Mail/IMAP-$(mail-user)\@$(mail-host)/.OfflineCache ;  }
mail-offline-cache-cd(){ cd $(mail-offline-cache) ; } 
mail-offline-cache-ls(){ ls -l $(mail-offline-cache) ; } 
mail-offline-cache-rm(){ rm $(mail-offline-cache)/* ; } 

mail-tmp(){
  echo /tmp/env/${FUNCNAME/-*/}
}

mail-user(){ echo $USER ; }
mail-host(){ echo hep1.phys.ntu.edu.tw ; }

mail-ports(){
  echo 993,143   
}


mail-run(){

  local msg="=== $FUNCNAME :"
  local iwd=$PWD
  local tmp=$(mail-tmp) && mkdir -p $tmp
  cd $tmp


  local mail=/Applications/Mail.app/Contents/MacOS/Mail
  [ ! -f $mail ] && echo $msg ERROR no $mail && return 1

  $mail -LogSocketErrors YES -LogActivityOnHost $(mail-host) -LogActivityOnPort $(mail-ports) -LogIMAPErrors YES

   

}

mail-cd(){
  cd $(mail-tmp)
}


