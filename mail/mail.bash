
mail-env(){
  elocal-
}

mail-usage(){

  cat << EOU

   Debugging mac Mail.app 

    mail-tmp : $(mail-tmp)
    mail-host : $(mail-host)
    mail-ports : $(mail-ports)

    mail-cd  :
    mail-run :
        exit Mail.app then run this to start it up with a live view of error messages 


EOU

}


mail-tmp(){
  echo /tmp/env/${FUNCNAME/-*/}
}

mail-host(){
  echo hep1.phys.ntu.edu.tw
}

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


