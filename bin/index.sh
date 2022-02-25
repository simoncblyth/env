#!/bin/bash 

${IPYTHON:-ipython} $(which index.py) 
open http://localhost/env/presentation/index.html


