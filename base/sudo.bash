sudo-source(){ echo $BASH_SOURCE ; }
sudo-usage(){
   cat << EOU

    Exploring alternative to the problematic SUDO variable approach, 

       sudo-source : $(sudo-source)
       sudo-- <args>
             
           sudo-- trac- \;  trac-admin- $*
           sudo-- trac- \; TRAC_INSTANCE=newtest trac-admin- permission list
   
      Depending on the frequency of usage thru "sudo bash -c" put the
      "shim" in various places, eg :
      
           tracperm--  TRAC_INSTANCE=newtest tracperm-prepare
           trac-- TRAC_INSTANCE=newtest trac-admin- permission list
           TRAC_INSTANCE=newtest trac-admin-- permission list
    
                               
EOU

}

sudo-env(){ echo -n ; }
sudo--(){  
   sudo bash -c "export ENV_HOME=$ENV_HOME ; . $ENV_HOME/env.bash ; env- ; $* "
}
