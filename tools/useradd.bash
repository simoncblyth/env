# === func-gen- : tools/useradd fgp tools/useradd.bash fgn useradd fgh tools src base/func.bash
useradd-source(){   echo ${BASH_SOURCE} ; }
useradd-edir(){ echo $(dirname $(useradd-source)) ; }
useradd-ecd(){  cd $(useradd-edir); }
useradd-dir(){  echo $LOCAL_BASE/env/tools/useradd ; }
useradd-cd(){   cd $(useradd-dir); }
useradd-vi(){   vi $(useradd-source) ; }
useradd-env(){  elocal- ; }
useradd-usage(){ cat << EOU

useradd
==========

* https://linuxize.com/post/how-to-create-users-in-linux-using-the-useradd-command/

Create new user
-----------------

::

    #name=noopticks
    name=charles
    sudo useradd $name

To enable ssh access had to chanhe config file and restart
------------------------------------------------------------

N::

    sudo su                           # attain root
    vi /etc/ssh/sshd_config           # add to AllowUsers
    systemctl restart sshd.service    # 

Now can connet from laptop with::
 
    ssh C


Must set the password to be able to login to the new account
---------------------------------------------------------------

::

    #name=noopticks
    name=charles
    sudo passwd $name

    [blyth@localhost ~]$ sudo passwd noopticks
    [sudo] password for blyth: 
    Changing password for user noopticks.
    New password: 
    Retype new password: 
    passwd: all authentication tokens updated successfully.
    [blyth@localhost ~]$ 


SSH setup
------------

Normal tunnel::

    epsilon:~ blyth$ t tun
    tun () 
    { 
        local mbip="internal.ip.address.that.works.from.L7.gateway.to.target.machine";
        local cmd="ssh -nNTv -L2001:$mbip:22 L7";
        echo $FUNCNAME : opening tunnel to precision via lxplus, use it in another tab with : ssh P;
        echo $cmd;
        eval $cmd
    }

The tunnel works for the new user, albeit via password::

    epsilon:~ blyth$ ssh N
    noopticks@127.0.0.1's password: 
    [noopticks@localhost ~]$ 

Plant the public key for passwordless, thru the tunnel::

    epsilon:~ blyth$ ssh--putkey N
    noopticks@127.0.0.1's password: 
    noopticks@127.0.0.1's password: 
    noopticks@127.0.0.1's password: 
    epsilon:~ blyth$ 

Passwordless works::

    epsilon:~ blyth$ ssh N 
    Last login: Fri Mar  5 02:33:31 2021 from lxslc704.ihep.ac.cn
    [noopticks@localhost ~]$ 






EOU
}
useradd-get(){
   local dir=$(dirname $(useradd-dir)) &&  mkdir -p $dir && cd $dir

}
