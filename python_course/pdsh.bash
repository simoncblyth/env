pdsh-vi(){ vi $BASH_SOURCE ; }
pdsh-env(){ echo -n ; }
pdsh-usage(){ cat << EOU
Python Data Science Handbook
============================

~/env/python_course/

https://nbviewer.org/

https://nbviewer.org/github/jakevdp/WhirlwindTourOfPython/blob/master/Index.ipynb

https://nbviewer.org/github/jakevdp/

https://nbviewer.org/github/jakevdp/PythonDataScienceHandbook/tree/master/notebooks/

https://code.ihep.ac.cn/blyth/PythonDataScienceHandbook   

EOU
}

pdsh-cd(){
   cd $HOME/pdsh 
}

pdsh-go(){
   pdsh-cd
   echo "View in browser at : open http://localhost:8000/PythonDataScienceHandbook/index.html "
   python -m http.server
}




