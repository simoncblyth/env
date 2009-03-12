cronline-src(){    echo base/cronline.bash ; }
cronline-source(){ echo $(env-home)/$(cronline-src) ; }
cronline-url(){    echo  $(env-url)/$(cronline-src) ; }
cronline-vi(){     vi $(cronline-source) ; }

cronline-usage(){
cat << EOU

   A streamlined and generalized development on cron.bash
   ... to generate the cron command lines to run generalized functions

      cronline-environment : $(cronline-environment)
      cronline-time        : $(cronline-time)
          cron time to run the command specification, influence via CRONLINE_* env vars eg
          CRONLINE_DELTA=15 cronline-time  
          NB the minutes are modulo 60, without carry over to the next hour


      cronline-logdir      : $(cronline-logdir)




   Usage example , to setup the crontabs for doing backup on grid1 by the dayabaysoft user
   ...  prepare the cronlines to be added to the command line.

        1) the backup :
                           cronline-cronline  scm-backup-all 

        2) 25mins later the rsyncing offbox

         CRONLINE_DELTA=25 cronline-cronline " scm-backup-rsync ; scm-backup-mail "
                   NB usage of quotes when a semi-colon is needed

EOU

}

cronline-env(){
   elocal-
}

cronline-environment(){
  echo "export HOME=$HOME ; export NODE=$LOCAL_NODE ; export MAILTO=$(env-email) ; export ENV_HOME=$ENV_HOME ; . $ENV_HOME/env.bash ; env- "
}

cronline-logdir(){ echo $(local-var-base)/log ; }
cronline-delta(){  echo ${CRONLINE_DELTA:-0}    ; }  # 0 .. CRONLINE_MINUTE - CRONLINE_DELTA  
cronline-minute(){ echo ${CRONLINE_MINUTE:-30} ; }   # 0-59 
cronline-hour(){   echo ${CRONLINE_HOUR:-04}   ; }   # 0-23 

cronline-time(){  
      local        delta=$(cronline-delta)
      local       minute=$(cronline-minute)
      local         hour=$(cronline-hour)
      local day_of_month="*"  # (1 - 31)
      local        month="*"  # (1 - 12)
      local  day_of_week="*"  # (0 - 7) (Sunday=0 or 7)
cat << EOT
  $((( $minute + $delta ) % 60)) $hour $day_of_month $month $day_of_week
EOT
}



cronline-cmd(){
     local cmd1=${1:-cronline-cmd-expects-arguments}
     cat << EOC
( $(cronline-environment) ; $* ) > $(cronline-logdir)/$cmd1.log 2>&1 
EOC

}

cronline-cronline(){
cat << EOC
   $(cronline-time)  $(cronline-cmd $*)  
EOC
}









