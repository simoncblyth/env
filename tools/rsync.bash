rsync-source(){ echo $BASH_SOURCE ; }
rsync-vi(){ vi $(rsync-source) ; }
rsync-env(){ elocal- ; }
rsync-usage(){ cat << EOU

rsync 
=======

See http://localhost/w/wiki/rsync


Backup Bouncer
----------------

* http://romej.com/2016/04/rsync-backups-on-os-x/

* https://github.com/n8gray/Backup-Bouncer
* https://github.com/mbaltaks/Backup-Bouncer




Lack of preservation of HFS+ compression in macports rsync 3.1.3 (vs 3.1.2)
------------------------------------------------------------------------------

::

    epsilon:patches blyth$ head -25 hfs-compression.diff
    This patch adds support for HFS+ compression.

    Written by Mike Bombich.  Taken from http://www.bombich.com/rsync.html

    Modified by Wayne to fix some issues and tweak the implementation a bit.
    This compiles on OS X and passes the testsuite, but otherwise UNTESTED!

    To use this patch, run these commands for a successful build:

        patch -p1 <patches/fileflags.diff
        patch -p1 <patches/crtimes.diff
        patch -p1 <patches/hfs-compression.diff
        ./prepare-source
        ./configure
        make

    TODO:
     - Should rsync try to treat the compressed data as file data and use the
       rsync algorithm on the data transfer?


macports Portfile
~~~~~~~~~~~~~~~~~~~

* https://github.com/macports/macports-ports/blob/master/net/rsync/Portfile

Comment from Portfile::

    # these come from http://rsync.samba.org/ftp/rsync/rsync-patches-3.1.3.tar.gz
    # and need to be updated with each release
    # hfs-compression.diff is marked by upstream as broken as of 3.1.3
    # removed: patch-hfs-compression.diff patch-hfs-compression-options.diff
    patchfiles          patch-fileflags.diff \
                        patch-crtimes.diff \
                        patch-acls-unpack-error.diff


rsync mailing list
~~~~~~~~~~~~~~~~~~~~~

* https://www.mail-archive.com/search?l=rsync%40lists.samba.org&q=hfs+compression&x=0&y=0





options
--------

::

   -a, --archive      archive mode; same as -rlptgoD (no -H)

   -r, --recursive    recurse into directories
   -l, --links        copy symlinks as symlinks
   -p, --perms        preserve permissions
   -t, --times        preserve times
   -g, --group        preserve group
   -o, --owner        preserve owner (super-user only)
   -D                 same as --devices --specials
    --devices         preserve device files (super-user only)
    --specials        preserve special files


   -z, --compress     compress file data during the transfer
                      (some forum posts not appropriate for local transfers)

   --inplace          update destination files in-place 
                      (some forum posts recommend for local transfers)

   --modify-window=N  compare mod-times with reduced accuracy
   --size-only        skip files that match in size 



mac clone relevant
--------------------

::

    --archive
      equivalent to -rlptgoD : 
     
      Note that -a does not preserve hardlinks, because finding multiply-linked
      files is expensive.  You must separately specify -H.  Note  also that for
      backward compatibility, -a currently does not imply the --fileflags option.
        

    --fileflags

      This  option  causes rsync to update the file-flags to be the
      same as the source files and directories (if your OS supports the chflags(2)
      system call).   Some flags can only be altered by the super-user and some might
      only be unset below a certain secure-level  (usually  single-user  mode).  It
      will  not  make  files  alterable  that  are  set  to  immutable  on  the
      receiver.  To do that, see --force-change, --force-uchange, and
      --force-schange.

   -A, --acls

      This option causes rsync to update the destination ACLs to be the same as
      the source ACLs.  The option also implies --perms.

      The  source  and destination systems must have compatible ACL entries for
      this option to work properly.  See the --fake-super option for a way to backup
      and restore ACLs that are not compatible.


    --hfs-compression

      This  option causes rsync to preserve HFS+ compression if the
      destination filesystem supports it.  If the destination does not support it,
      rsync will exit with an error.

      Filesystem compression was introduced to HFS+ in Mac OS 10.6. A file
      that is compressed has no data in its data  fork.  Rather,  the  compressed
      data is stored in an extended attribute named com.apple.decmpfs and a file
      flag is set to indicate that the file is compressed (UF_COMPRESSED). HFS+
      decompresses this data "on-the-fly" and presents it to the operating system as
      a normal file. Normal attempts to copy compressed files (e.g. in the
      Finder, via cp, ditto, etc.) will copy the file's decompressed contents, remove
      the UF_COMPRESSED file flag, and discard the com.apple.decmpfs extended
      attribute. This option will preserve the data in the com.apple.decmpfs extended
      attribute and ignore the synthesized data in the file contents.

      This option implies both --fileflags and (--xattrs).

      -X, --xattrs

      This option causes rsync to update the destination extended
      attributes to be the same as the source ones.

      For systems that support extended-attribute namespaces, a copy being
      done by a super-user copies all namespaces except system.*.  A normal user
      only  copies  the  user.*  namespace.   To  be able to backup and restore
      non-user namespaces as a normal user, see the --fake-super option.

      Note that this option does not copy rsyncs special xattr values (e.g.
      those used by --fake-super) unless you repeat the option (e.g. -XX).  This
      "copy all xattrs" mode cannot be used with --fake-super.

      --protect-decmpfs

      The com.apple.decmpfs extended attribute is hidden by default
      from list/get xattr calls, therefore normal attempts to copy compressed files
      will functionally decompress those files. While this is desirable behavior when
      copying files to filesystems that do not support HFS+ compression, it has
      serious performance and capacity impacts when backing up or restoring the Mac
      OS X filesystem.

      This option will transfer the com.apple.decmpfs extended
      attribute regardless of support on the destination. If a source file is com-
      pressed and an existing file on the destination is not compressed, the data
      fork of the destination file will be truncated and the
      com.apple.decmpfs xattr will be transferred instead. Note that compressed files
      will not be readable to the operating system of the destination if that
      operating system does not support HFS+ compression. Once restored (with or
      without this option) to an operating system that supports HFS+ compression,
      however, these files will be accessible as usual.

      This option implies --fileflags and --xattrs.


      -N, --crtimes               preserve create times (newness)


      -x, --one-file-systema

       This  tells rsync to avoid crossing a filesystem boundary when
       recursing.  This does not limit the user's ability to specify items to copy
       from multiple filesystems, just rsync's recursion through the hierarchy of each
       directory that the user specified, and also the  analogous recursion  on  the
       receiving side during deletion.  Also keep in mind that rsync treats a "bind"
       mount to the same device as being on the same filesystem.

       If this option is repeated, rsync omits all mount-point
       directories from the copy.  Otherwise, it includes  an  empty  directory  at
       each mount-point  it  encounters (using the attributes of the mounted directory
       because those of the underlying mount-point directory are inac- cessible).

       If rsync has been told to collapse symlinks (via --copy-links or
       --copy-unsafe-links), a symlink to  a  directory  on  another  device  is
       treated like a mount-point.  Symlinks to non-directories are unaffected by this
       option.


        --delete

        This tells rsync to delete extraneous files from the receiving
        side (ones that aren't on the sending side), but only for  the  directories
        that are being  synchronized.  You must have asked rsync to send the whole
        directory (e.g. "dir" or "dir/") without using a wildcard for the directory's
        contents (e.g. "dir/*") since the wildcard is expanded by the shell and rsync
        thus gets a request to  transfer  individual files, not the files' parent
        directory.  Files that are excluded from the transfer are also excluded from
        being deleted unless you use the --delete-excluded option or mark the rules as
        only matching on the sending side (see the include/exclude modifiers  in  the
        FILTER  RULES section).

        Prior  to  rsync  2.6.7, this option would have no effect unless
        --recursive was enabled.  Beginning with 2.6.7, deletions will also occur when
        --dirs (-d) is enabled, but only for directories whose contents are being
        copied.

        This option can be dangerous if used incorrectly!  It is a very
        good idea to first try a run using the --dry-run option (-n) to  see  what
        files are going to be deleted.

        If the  sending side detects any I/O errors, then the deletion
        of any files at the destination will be automatically disabled. This is to
        prevent temporary filesystem failures (such as NFS errors) on the sending side
        from causing a massive deletion of files on the destination.  You can
        override this with the --ignore-errors option.

        The  --delete  option  may  be combined with one of the
        --delete-WHEN options without conflict, as well as --delete-excluded.  However,
        if none of the --delete-WHEN options are specified, rsync will choose the
        --delete-during algorithm when talking to rsync 3.0.0 or newer, and the
        --delete-before algorithm when talking to an older rsync.  See also
        --delete-delay and --delete-after.

      --inplace

      This option changes how rsync transfers a file when its data
      needs to be updated: instead of the default method of creating a new copy of
      the file and moving it into place when it is complete, rsync instead writes the
      updated data directly to the destination file.

      This has several effects:

      o Hard links are not broken.  This means the new data will
        be visible through other hard links to the  destination  file.   Moreover,
        attempts  to copy differing source files onto a multiply-linked destination
        file will result in a "tug of war" with the destination data changing back and
        forth.

      o In-use binaries cannot be updated (either the OS will
        prevent this from happening, or binaries that attempt to swap-in  their  data
        will misbehave or crash).

      o The file's data will be in an inconsistent state during
        the transfer and will be left that way if the transfer is interrupted or if an
        update fails.

      o A file that rsync cannot write to cannot be updated. While
        a super user can update any file, a normal  user  needs  to  be  granted write
        permission for the open of the file for writing to be successful.

      o The efficiency of rsync's delta-transfer algorithm may be
        reduced if some data in the destination file is overwritten before it can be
        copied to a position later in the file.  This does not apply if you use
        --backup, since rsync is smart enough to use the  backup file as the basis file
        for the transfer.

      WARNING:  you  should not use this option to update files that
      are being accessed by others, so be careful when choosing to use this for a
      copy.

      This option is useful for transferring large files with
      block-based changes or appended data, and also on systems that are disk bound,
      not network bound.  It can also help keep a copy-on-write filesystem snapshot
      from diverging the entire contents of a file that only has minor changes.

      The option implies  --partial  (since  an  interrupted  transfer
      does  not  delete  the  file),  but  conflicts  with  --partial-dir  and
      --delay-updates.  Prior to rsync 2.6.4 --inplace was also incompatible with
      --compare-dest and --link-dest.




logging control
----------------

::

    rsync --itemize-changes --recursive /usr/local/home/_build/ /tmp/rtest/_build  

::

    delta:home blyth$ rsync --info=help
    Use OPT or OPT1 for level 1 output, OPT2 for level 2, etc.; OPT0 silences.

    BACKUP     Mention files backed up
    COPY       Mention files copied locally on the receiving side
    DEL        Mention deletions on the receiving side
    FLIST      Mention file-list receiving/sending (levels 1-2)
    MISC       Mention miscellaneous information (levels 1-2)
    MOUNT      Mention mounts that were found or skipped
    NAME       Mention 1) updated file/dir names, 2) unchanged names
    PROGRESS   Mention 1) per-file progress or 2) total transfer progress
    REMOVE     Mention files removed on the sending side
    SKIP       Mention files that are skipped due to options used
    STATS      Mention statistics at end of run (levels 1-3)
    SYMSAFE    Mention symlinks that are unsafe

    ALL        Set all --info options (e.g. all4)
    NONE       Silence all --info options (same as all0)
    HELP       Output this help message

    Options added for each increase in verbose level:
    1) COPY,DEL,FLIST,MISC,NAME,STATS,SYMSAFE
    2) BACKUP,MISC2,MOUNT,NAME2,REMOVE,SKIP
    delta:home blyth$ 


man rsyncd.conf
----------------





rsync and symbolic links
-------------------------

See home/bin/rsynclog.py 




rsync to FAT32 (MS-DOS) target ?
-----------------------------------

* https://serverfault.com/questions/54949/how-can-i-use-rsync-with-a-fat-file-system

I would recommend reformatting to a linux fs if you possibly can. As mentioned,
FAT has relatively low file size limits and might not handle permissions and
ownership quite right. More importantly, FAT doesn't track modification times
on files as precisely as, say ext3 (FAT is only precise to within a 2 second
window). This leads to particularly nasty behavior with rsync as it will
sometimes decide that the original files is newer or older than the backup file
by enough that it needs to re-copy the data or at least re-check the hashes.
All in all, it makes for very poor performance on backups. If you must stick
with FAT, look into rsync's --size-only and --modify-window flags as
workarounds.


My Experience:

* using --modify-window=1 succeeds to make update transfers for a FAT32 formatted 
  USB stick fast : so the problem does appear to be modification 
  time precision differences between file systems

EOU
}

rsync-ver(){ echo 3.1.3 ; }
rsync-nam(){ echo rsync-$(rsync-ver) ; }
rsync-pam(){ echo rsync-patches-$(rsync-ver) ; }
rsync-pfx(){ echo /tmp/$USER/env/rsync/$(rsync-nam).install ; }
rsync-dir(){ echo /tmp/$USER/env/rsync/$(rsync-nam) ; }
rsync-cd(){ cd $(rsync-dir) ; }

rsync-info(){ cat << EOI

   rsync-ver : $(rsync-ver)
   rsync-nam : $(rsync-nam)
   rsync-pam : $(rsync-pam)
   rsync-pfx : $(rsync-pfx)
   rsync-dir : $(rsync-dir)

   # see ~/home/wiki/rsync.rst  (old Trac wiki notes)

EOI
}


rsync-get(){

  local dir=$(dirname $(rsync-dir))
  mkdir -p $dir 
  cd $dir

  local nam=$(rsync-nam)
  local pam=$(rsync-pam)

  local tgz=$nam.tar.gz
  [ ! -f $tgz ] && curl -O http://rsync.samba.org/ftp/rsync/$tgz
  [ ! -d $nam ] && tar -xzvf $tgz

  local pgz=$pam.tar.gz
  [ ! -f $pgz ] && curl -O http://rsync.samba.org/ftp/rsync/$pgz
  [ ! -d $nam/patches ] && tar -zxvf $pgz

}


rsync-patch(){
   local nonce=$FUNCNAME-DONE

   rsync-cd
   
   if [ ! -f $nonce ]; then
     patch -p1 <patches/fileflags.diff
     patch -p1 <patches/crtimes.diff
     touch $nonce
   else
     echo already patched
     ls -l $nonce
   fi
   
}



rsync-build(){

   rsync-cd

   local pfx=$(rsync-pfx)

  ./prepare-source   
  #   make: Nothing to be done for `conf'.
  
  ./configure --prefix=$pfx
#
# checking attr/xattr.h usability... no
# checking attr/xattr.h presence... no
# checking for attr/xattr.h... no
# checking sys/xattr.h usability... yes
# checking sys/xattr.h presence... yes
# checking for sys/xattr.h... yes
# checking sys/extattr.h usability... no
# checking sys/extattr.h presence... no
# checking for sys/extattr.h... no
# checking whether to support extended attributes... Using OS X xattrs
# configure.sh: creating ./config.status
# config.status: creating Makefile
#

  make
  #sudo make install


}

rsync-install(){
  sudo make install
}

rsync-uninstall(){
   local pfx=$(rsync-pfx)
   local cmd="sudo rm -r $pfx"
   echo $cmd
   eval $cmd
}


#rsync-wipe(){
#	rsync-var
#	rm -r $tmp
#    rsync-unvar  
#}


