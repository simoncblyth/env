# === func-gen- : tools/tcpdump fgp tools/tcpdump.bash fgn tcpdump fgh tools
tcpdump-src(){      echo tools/tcpdump.bash ; }
tcpdump-source(){   echo ${BASH_SOURCE:-$(env-home)/$(tcpdump-src)} ; }
tcpdump-vi(){       vi $(tcpdump-source) ; }
tcpdump-env(){      elocal- ; }
tcpdump-usage(){ cat << EOU

* https://danielmiessler.com/study/tcpdump/

::

    simon:env blyth$ tcpdump-w
    Password:
    tcpdump: listening on en5, link-type EN10MB (Ethernet), capture size 65535 bytes
    10000 packets captured
    10005 packets received by filter
    0 packets dropped by kernel
    simon:env blyth$ 
    simon:env blyth$ 
    simon:env blyth$ du -h /tmp/data.cap 
    4.2M    /tmp/data.cap
    simon:env blyth$ 


EOU
}
tcpdump-dir(){ echo $(local-base)/env/tools/tools-tcpdump ; }
tcpdump-cd(){  cd $(tcpdump-dir); }
tcpdump-mate(){ mate $(tcpdump-dir) ; }
tcpdump-get(){
   local dir=$(dirname $(tcpdump-dir)) &&  mkdir -p $dir && cd $dir

}

tcpdump-cap(){ echo /tmp/data.cap ; }
tcpdump-w()
{
    local interface=${1:-en5}
    local cap=$(tcpdump-cap)
    local npacket=10000

    sudo tcpdump \
            -i $interface \
            -na \
            -s0 \
            -w $cap \
            -c $npacket  

    # -w $cap        write to cap file
    # -c $npacket    limit number of packets captured
    # -s0            snaplength, 0 means everything 
    # -n 
}
tcpdump-r()
{
   tcpdump -r $(tcpdump-cap)
}




