# === func-gen- : osx/tmutil/tmutil fgp osx/tmutil/tmutil.bash fgn tmutil fgh osx/tmutil
tmutil-src(){      echo osx/tmutil/tmutil.bash ; }
tmutil-source(){   echo ${BASH_SOURCE:-$(env-home)/$(tmutil-src)} ; }
tmutil-vi(){       vi $(tmutil-source) ; }
tmutil-env(){      elocal- ; }
tmutil-usage(){ cat << EOU

Time Machine from Commandline
===============================

::

   man tmutil


* https://www.macworld.com/article/2033804/control-time-machine-from-the-command-line.html


Running listbackups mounted /Volumes/Time Machine Backups

::

    delta:env blyth$ tmutil listbackups
    /Volumes/Time Machine Backups/Backups.backupdb/delta/2018-01-08-195921


With files being visible from Finder, /usr/local and /var/scm are not visible from Finder 
but it is from Terminal::


    delta:env blyth$ l /Volumes/Time\ Machine\ Backups/Backups.backupdb/delta/2018-01-08-195921/Delta/usr/local/opticks/
    total 256
    drwxr-xr-x@  2 blyth  staff   13838 Dec 17 16:01 lib
    drwxr-xr-x@ 24 blyth  staff    1122 Dec 14 21:07 build
    drwxr-xr-x@  8 blyth  staff     306 Nov 30 17:46 opticksdata
    drwxr-xr-x@  4 blyth  staff     170 Nov 29 13:50 geocache
    drwxr-xr-x@ 20 blyth  staff     680 Sep 12 16:05 include
    drwxr-xr-x@  2 blyth  staff     680 Sep 12 14:32 bin
    drwxr-xr-x@ 15 blyth  staff     782 Sep  4 18:10 gl
    drwxr-xr-x@ 20 blyth  staff     714 Jun 14  2017 externals
    drwxr-xr-x@  5 blyth  staff     170 Jun 14  2017 installcache
    -rw-r--r--@  1 blyth  staff  127384 Jun 14  2017 opticks-externals-install.txt

    delta:~ blyth$ l /Volumes/Time\ Machine\ Backups/Backups.backupdb/delta/2018-01-08-195921/Delta/var/scm/mercurial/
    total 0
    drwxr-xr-x@ 3 blyth  staff  102 Apr 29  2015 workflow
    drwxr-xr-x@ 3 blyth  staff  102 Sep  8  2014 tracdev
    drwxr-xr-x@ 3 blyth  staff  102 Sep  4  2014 heprez
    drwxr-xr-x@ 3 blyth  staff  102 Sep  1  2014 env
    drwxr-xr-x@ 3 blyth  staff  102 Jul 21  2014 env.jul21

    delta:~ blyth$ l /Volumes/Time\ Machine\ Backups/Backups.backupdb/delta/2018-01-08-195921/Delta/var/scm/subversion/
    total 0
    drwxr-xr-x@ 6 blyth  staff  272 Apr 29  2015 workflow
    drwxr-xr-x@ 6 blyth  staff  272 Sep  4  2014 tracdev
    drwxr-xr-x@ 6 blyth  staff  272 Sep  4  2014 heprez
    drwxr-xr-x@ 6 blyth  staff  272 Jul 31  2014 env
    delta:~ blyth$ 




With is a friendly view of the sparsebundle content:: 

    delta:env blyth$ l /Volumes/blyth/
    total 0
    drwxrwxrwx@ 1 blyth  staff  296 Jan  9 19:28 delta.sparsebundle

    delta:env blyth$ l /Volumes/blyth/delta.sparsebundle/
    total 48
    drwxrwxrwx  1 blyth  staff  660508 Jan  8 20:54 bands
    -rwxrwxrwx  1 blyth  staff     515 Jan  8 20:53 com.apple.TimeMachine.SnapshotHistory.plist
    -rwxrwxrwx  1 blyth  staff     516 Jan  8 20:53 com.apple.TimeMachine.MachineID.bckup
    -rwxrwxrwx  1 blyth  staff     516 Jan  8 20:53 com.apple.TimeMachine.MachineID.plist
    -rwxrwxrwx  1 blyth  staff    1277 Jan  8 20:00 com.apple.TimeMachine.Results.plist
    -rwxrwxrwx  1 blyth  staff     499 Jan  8 15:22 Info.bckup
    -rwxrwxrwx  1 blyth  staff     499 Jan  8 15:22 Info.plist
    -rwxrwxrwx  1 blyth  staff       0 Jan  8 15:22 token
    delta:env blyth$ 



EOU
}
tmutil-dir(){ echo $(local-base)/env/osx/tmutil/osx/tmutil-tmutil ; }
tmutil-cd(){  cd $(tmutil-dir); }
tmutil-mate(){ mate $(tmutil-dir) ; }
tmutil-get(){
   local dir=$(dirname $(tmutil-dir)) &&  mkdir -p $dir && cd $dir

}
