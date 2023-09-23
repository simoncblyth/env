#!/bin/bash 



${IPYTHON:-ipython} --pdb -i $(which index.py) 


open http://localhost/env/presentation/index.html


