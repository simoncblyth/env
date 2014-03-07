#!/usr/bin/env python
"""

NB Using virtualenv /usr/local/env/chroma_env/bin/python in shebang 
does not work as there is more to "chroma-" than just the picking the 
python



Native 2880,1800
Half   1440,900 

"""
import os
from chroma.camera import Camera
from chroma.tools import enable_debug_on_crash
import chroma.loader
from chroma.log import logger, logging
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    enable_debug_on_crash()

    path = os.environ['DAE_NAME']
    geometry = chroma.loader.load_geometry_from_string(path)

    camera_kwargs = {}
    size = [1440,900]
    #size = [1024,576]

    camera = Camera(geometry, size, **camera_kwargs)
    camera._run()





