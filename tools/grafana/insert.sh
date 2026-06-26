#!/usr/bin/env bash 

usage(){ cat << EOU
tools/grafana/insert.sh
========================

The sqlite3 DB read by grafana needs to be owned by "grafana" user.
This script uses sudo to create a mockup DB owned by "grafana"
using mockup.sql

TODO: more realistic mockup.sql - to help with deciding on suitable schema
TODO: get sreport/ranges to sqlite3 conversion working

See ~/j/opticks_monitoring/nightly_tests_sql_schema_for_grafana_consumption.rst

EOU
}


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


