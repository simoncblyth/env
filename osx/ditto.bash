# === func-gen- : osx/ditto fgp osx/ditto.bash fgn ditto fgh osx
ditto-src(){      echo osx/ditto.bash ; }
ditto-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ditto-src)} ; }
ditto-vi(){       vi $(ditto-source) ; }
ditto-env(){      elocal- ; }
ditto-usage(){ cat << EOU

ditto
========

Extracts from *man ditto*


EXAMPLES
     The command:
           ditto src_directory dst_directory
     copies the contents of src_directory into dst_directory, creating
     dst_directory if it does not already exist.

BUGS
     ditto doesn't copy directories into directories in the same way as cp(1).  In particular,
           ditto foo bar
     will copy the contents of foo into bar, whereas
           cp -r foo bar
     copies foo itself into bar. Though this is not a bug, some may consider
     this bug-like behavior.  --keepParent for non-archive copies will eventually
     alleviate this problem.


 --hfsCompression
      When copying files or extracting content from an archive, if the
      destination is an HFS+ volume that supports compression, all the content will
      be compressed if appropriate. This is only supported on Mac OS X 10.6 or later,
      and is only intended to be used in installation and backup scenarios that
      involve system files. Since files using HFS+ compression are not readable on
      versions of Mac OS X earlier than 10.6, this flag should not be used when
      dealing with non-system files or other user-generated content that will be used
      on a version of Mac OS X earlier than 10.6.

 --nohfsCompression
      Do not compress files with HFS+ compression when copying or extracting
      content from an archive unless the content is already compressed with HFS+
      compression.  This flag is only supported on Mac OS X 10.6 or later.
      --nohfsCompression is the default.

 --preserveHFSCompression
      When copying files to an HFS+ volume that supports compression, ditto
      will preserve the compression of any source files that were using HFS+
      compression.  This flag is only supported on Mac OS X 10.6 or later.
      --preserveHFSCompression is the default.

 --nopreserveHFSCompression
      Do not preserve HFS+ compression when copying files that are already
      compressed with HFS+ compression. This is only supported on Mac OS X 10.6 or
      later.



EOU
}
ditto-dir(){ echo $(local-base)/env/osx/osx-ditto ; }
ditto-cd(){  cd $(ditto-dir); }
ditto-mate(){ mate $(ditto-dir) ; }
ditto-get(){
   local dir=$(dirname $(ditto-dir)) &&  mkdir -p $dir && cd $dir

}
