import os
import sys

fonts_dir = sys.argv[1]

for subdir, dirs, files in os.walk(fonts_dir):
    for d in dirs:
        ttf_dir = os.path.join(subdir, d)
        out_dir = os.path.join(sys.argv[2], d)
        print("Generating data for font {}".format(d))
        os.system("trdg -c {} -w 1 -f 100 -fd {} --output_dir {}".format(sys.argv[3], ttf_dir, out_dir))
