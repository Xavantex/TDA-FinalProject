from numpy import loadtxt, asarray
from pathlib import Path

def loadNatural(p: Path):
    # data/outdoorBox_met/box_met_###/box_met_###.xyz
    return loadtxt(p, delimiter=' ')


import struct

def __convert(x_s, y_s, z_s):

    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, z


def loadUrban(p: Path):

    # data/urban2012-01-08/velodyne_sync/132603xxxxxxxxxx.bin
    f_bin = open(p, "rb")

    hits = []

    while True:

        x_str = f_bin.read(2)
        if x_str == b'': # eof
            break

        x = struct.unpack('<H', x_str)[0]
        y = struct.unpack('<H', f_bin.read(2))[0]
        z = struct.unpack('<H', f_bin.read(2))[0]
        i = struct.unpack('B', f_bin.read(1))[0]
        l = struct.unpack('B', f_bin.read(1))[0]

        x, y, z = __convert(x, y, z)

        # Load in homogenous
        hits += [[x, y, z]]

    f_bin.close()
    return asarray(hits)