from numpy import loadtxt, asarray
from pathlib import Path

from wget import download
from tarfile import open as gzopen
from zipfile import ZipFile as zipopen



# Return default download path
def naturalPath():
    return Path(__file__).parent / 'data/natural/box_met'


# Download file if it is nonexistant
def __downloadNatural(p: Path):
    url = "ftp://asrl3.utias.utoronto.ca/3dmap_dataset/box_met/box_met.zip"
    dataP = Path(__file__).parent / "data"
    gzP = dataP / "natural.zip"
    if not p.is_file():
        download(url, str(gzP))
        with zipopen(str(gzP), "r") as file:
            file.extractall(str(dataP / "natural"))
        gzP.unlink()
    

# Load the natural datapath into ndarray (x, 3)
def loadNatural(p: Path):

    __downloadNatural(p)

    # data/outdoorBox_met/box_met_###/box_met_###.xyz
    return loadtxt(p, delimiter=' ')




import struct

# Return default download path
def urbanPath():
    return Path(__file__).parent / 'data/urban/2012-01-08/velodyne_sync/'


# Download urban dataset if the file you look for is nonexistant
def __downloadUrban(p: Path):
    url = "https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/velodyne_data/2012-01-08_vel.tar.gz"
    dataP = Path(__file__).parent / "data"
    gzP = dataP / "urban.tar.gz"
    if not p.is_file():
        download(url, str(gzP))
        with gzopen(str(gzP), "r:gz") as file:
            file.extractall(str(dataP / "urban"))
        gzP.unlink()

# Convert to meters
def __convert(x_s, y_s, z_s):

    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, z

# Download urban dataset from urbpath into ndarry (x, 3)
def loadUrban(p: Path):

    __downloadUrban(p)

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