import shutil
import os

DATA_DIR = "/data/images/"

# current folder path
f = os.getcwd() + DATA_DIR

# parse the structure of images
sub_dirs = [y for x in os.walk(f) if (y := (x[0].split(f)[-1]))]
sub_structure = {x:[] for x in sub_dirs if "/" not in x}

for x in set(sub_dirs) - set(sub_structure.keys()):
    t, s = x.rsplit("/", 1)
    sub_structure[t] += [s]

# move all the images from nested folders to parent directory
for s, d in sub_structure.items():
    for sd in d:
        file_names = os.listdir("/".join([f, s, sd]))
        for file_name in file_names:
            src = "/".join([f[:-1], s, sd, file_name])
            dest = "/".join([f[:-1], s, file_name])
            print("\rMoving {0} -> {1}...".format(src, dest))
            shutil.move(src, dest)

        # remove parent folder
        shutil.rmtree("/".join([f[:-1], s, sd]))
