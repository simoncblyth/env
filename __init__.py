"""
   this directory is egglinked onto syspath via the setup.py ..


   python -c "import env ; print env.HOME   "

"""


def find_home():
    """
        While avoiding envvars ... 
    """
    import os
    path = os.path.dirname(__file__)
    if os.path.islink(path):
        return os.path.join( os.path.dirname(path), os.readlink( path ))
    else:
        return os.path.dirname(path) 

HOME = find_home() 
