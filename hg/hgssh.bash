# === func-gen- : hg/hgssh fgp hg/hgssh.bash fgn hgssh fgh hg
hgssh-src(){      echo hg/hgssh.bash ; }
hgssh-source(){   echo ${BASH_SOURCE:-$(env-home)/$(hgssh-src)} ; }
hgssh-vi(){       vi $(hgssh-source) ; }
hgssh-env(){      elocal- ; }
hgssh-usage(){ cat << EOU

Mercurial Over SSH
=====================

See also hgweb- hg-

For repos do not want to entrust to bitbucket, the 
simplicity of SSH access from remote linux box is
attractive.::

    delta:~ blyth$ hg clone ssh://blyth@cms01.phys.ntu.edu.tw//var/hg/repos/demo 
    Enter passphrase for key '/Users/blyth/.ssh/id_rsa': 
    remote: Scientific Linux CERN SLC release 4.8 (Beryllium)
    destination directory: demo
    no changes found
    updating to branch default
    0 files updated, 0 files merged, 0 files removed, 0 files unresolved
    delta:~ blyth$ 

SSH aliases work, NB the funny double-double slash::

    delta:~ blyth$ hg clone ssh://C//var/hg/repos/demo 
    remote: Scientific Linux CERN SLC release 4.8 (Beryllium)
    destination directory: demo
    no changes found
    updating to branch default
    0 files updated, 0 files merged, 0 files removed, 0 files unresolved
    delta:~ blyth$ 




EOU
}
hgssh-dir(){ echo $(local-base)/env/hg/hg-hgssh ; }
hgssh-cd(){  cd $(hgssh-dir); }
