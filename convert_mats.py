import glob
from scipy.io import loadmat
import struct

for p in glob.glob("*.mat"):
    f = loadmat(p)
    gts = f['groundTruth'][0]
    for i, gt in enumerate(gts):
        fo = open(p.replace('.mat', '_%s.dat' % str(i+1)), 'wb')
        regions = gt[0][0][0]
        boundaries = gt[0][0][1]
        fo.write(struct.pack('i' * (regions.size + 2), regions.shape[1], regions.shape[0], *regions.flatten().astype('int32')))
        fo.write(struct.pack('B' * (regions.size), *boundaries.flatten().astype('uint8')))
        fo.close()
