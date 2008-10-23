#!/bin/bash

# for the first time indexing
touch indexing_time.file
#/usr/local/bin/swish-e -c swish.conf -S prog

# for the incremental indexing
/usr/local/bin/swish-e -c swish.conf -S prog -N indexing_time.file -f index.tmp
/usr/local/bin/swish-e -M index.swish-e index.tmp index.new

# for cleanup
rm -f index.tmp*

# rename the new index
mv index.new       index.swish-e
mv index.new.array index.swish-e.array  
mv index.new.btree index.swish-e.btree
mv index.new.file  index.swish-e.file
mv index.new.prop  index.swish-e.prop
mv index.new.psort index.swish-e.psort
mv index.new.wdata index.swish-e.wdata
