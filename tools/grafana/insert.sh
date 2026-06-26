#!/usr/bin/env bash 


sql=mockup.sql
db=/usr/local/grafana/mock_data/test_metrics.db

vv="BASH_SOURCE sql db"
for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done

arg=${1:-$defarg}

if [ "${arg/clean}" != "$arg" ]; then 
    sudo rm -f $db
fi

if [ "${arg/insert}" != "$arg" ]; then 
    cat $sql | sudo su -s /bin/bash -c "sqlite3 $db" grafana
fi

if [ "${arg/WriteAheadLogging}" != "$arg" ]; then
    echo WAL model separates writes from reads - avoiding concurrent access locking between them
    sudo su -s /bin/bash -c "sqlite3 $db 'PRAGMA journal_mode=WAL;'" grafana
fi 


