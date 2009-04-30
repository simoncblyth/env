etc-vi(){  vi $BASH_SOURCE ; }
etc-env(){ echo -n ;}

etc-usage(){
   cat << EOU

   Extract fields from /etc/passwd for a user :

     etc-shell $USER : $(etc-shell $USER)
     etc-home  $USER : $(etc-home  $USER)

EOU
}

etc-shell(){ etc-user2field $1 7 ;}
etc-home(){  etc-user2field $1 6 ;}

etc-user2field(){ eval $(etc-awk- $*) ; }
etc-awk-(){
   local user=$1
   local field=$2 
   cat << EOS
awk -F":" ' \$1 == "$user" { print \$$field }' /etc/passwd
EOS
}
