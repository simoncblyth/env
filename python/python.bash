
PYTHON_BASE=$ENV_BASE/python
export PYTHON_ENV=$HOME/$PYTHON_BASE

python-(){       [ -r $PYTHON_ENV/python.bash ]           && . $PYTHON_ENV/python.bash ; }
ipython-(){      [ -r $PYTHON_ENV/ipython.bash ]           && . $PYTHON_ENV/ipython.bash ; }
