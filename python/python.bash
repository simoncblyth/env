
PYTHON_BASE=$ENV_BASE/python
export PYTHON_HOME=$HOME/$PYTHON_BASE

python-(){       [ -r $PYTHON_HOME/ python.bash ]           && . $PYTHON_HOME/python.bash ; }
ipython-(){      [ -r $PYTHON_HOME/ipython.bash ]           && . $PYTHON_HOME/ipython.bash ; }
