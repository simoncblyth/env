#!/usr/bin/env bash 


sql=mockup.sql
db=/usr/local/grafana/mock_data/test_metrics.db

vv="BASH_SOURCE sql db"
for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done

arg=${1:-$defarg}

if [ "${arg/insert}" != "$arg" ]; then 
    sudo rm -f $db
    cat $sql | sudo su -s /bin/bash -c "sqlite3 $db" grafana
fi



