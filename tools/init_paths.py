## MODIFIED FROM _init_paths.py
import os
import sys

def add_path(path):
    if path not in sys.path:
        # print('adding path {}'.format(path))
        sys.path.append(path)

PATH_CURRENT = os.path.abspath(os.path.dirname(__file__))
# print(PATH_CURRENT)

# get parent dir: https://stackoverflow.com/questions/2860153/how-do-i-get-the-parent-directory-in-python
from pathlib import Path
PATH_PARENT = Path(PATH_CURRENT).parent
#PATH_PARENT = os.path.abspath(os.path.dirname(__file__)+os.path.sep+os.pardir) # THIS DOESN'T WORK IN SOME ENVIRONMENTS
#print(PATH_PARENT)

PATH_LIB = os.path.join(PATH_PARENT, 'lib')
#print(PATH_LIB)
add_path(PATH_LIB)


PATH_MM = os.path.join(PATH_PARENT, 'lib/poseeval/py-motmetrics')
add_path(PATH_MM)


#print(sys.path)
