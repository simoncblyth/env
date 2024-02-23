leak-vi(){ vi $BASH_SOURCE ; }
leak-env(){ echo -n ; }
leak-usage(){ cat << EOU
leak.bash
==========

* https://www.freshblurbs.com/blog/2007/01/25/how-profile-memory-linux.html

* https://groups.google.com/g/comp.unix.solaris/c/Ed_hGn4-Eto?pli=1

We need to remember that traditional malloc() free()
calls are library calls and not system calls. That
means free() may not return the memory immediately to
the system.

[You can tune your malloc application to return memory
immediately to system by modifying its cache size to zero.
You can do that either on a system-wide basis or on a per
program basis. Refer man malloc]


* https://stackoverflow.com/questions/29684046/which-of-vsize-size-and-rss-should-be-used-in-memory-leak-detection



EOU
}
