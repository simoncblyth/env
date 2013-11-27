#!/bin/bash -l
[ -f Makefile ] && make distclean
rm -f demoif.{h,cpp}
rm -f demoifadaptor.{h,cpp}

