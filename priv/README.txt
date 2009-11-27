
Provides access to private config strings stored in 
the file pointed to via envvar ENV_PRIVATE_PATH 
via keys 

The format of the file is like the inside of a bash function, eg :

local AMQP_SERVER=wherever


Dependencies :
     pcre-devel    (sudo yum install pcre-devel )    
     glib 




Possible issues, 

  1)  if SELinux is in enforcing mode you may encounter :
{{{
[blyth@belle7 priv]$ make test
./lib/private_val: error while loading shared libraries: lib/libprivate.so:
cannot restore segment prot after reloc: Permission denied
make: *** [test] Error 127
}}}

    accompanied by entry in /var/log/audit/audit.log or /var/log/messages 

type=AVC msg=audit(1259304299.149:15424): avc:  denied  { execmod } for pid=23397 comm="private_val" path="/data1/env/local/env/home/priv/lib/libprivate.so" dev=dm-0 ino=185795124
scontext=user_u:system_r:unconfined_t:s0
tcontext=user_u:object_r:httpd_sys_content_t:s0 tclass=file

    Can be remedied by :
{{{
sudo chcon -t texrel_shlib_t $(env-home)/priv/lib/libprivate.so 
}}}



  2)   pcre version too old    (encounterd on grid1)

private.c: In function `parse_config':
private.c:64: `PCRE_INFO_NAMECOUNT' undeclared (first use in this function)
private.c:64: (Each undeclared identifier is reported only once
private.c:64: for each function it appears in.)
private.c:73: `PCRE_INFO_NAMETABLE' undeclared (first use in this function)
private.c:74: `PCRE_INFO_NAMEENTRYSIZE' undeclared (first use in this function)


    will need to install a more recent pcre

        pcre-
        pcre-build

            
              



