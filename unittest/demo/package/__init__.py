"""
package docstring in the __init__.py file
"""
import os
import sys
sys.path.append(os.path.join(os.environ['ENV_HOME'], "unittest/context" ) )


from context import whereami as whereami
whereami(globals())

