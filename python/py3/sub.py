#!/usr/bin/env python3

import subprocess 

if __name__ == '__main__':

     cmdline = "ls -1" 
     result = subprocess.run(cmdline.split(), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
     lines = result.stdout.split('\n')

     print(lines)





