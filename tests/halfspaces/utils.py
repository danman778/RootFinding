import os

def make_dir(dir,path = None):
    things = os.listdir(path)
    if path is None:
        path=""
    else:
        path += "/"
    if dir not in things:
        os.mkdir(path + dir)