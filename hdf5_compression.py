from enum import IntEnum


class FilterID(IntEnum):
    # https://portal.hdfgroup.org/display/support/Registered+Filters
    LZF = 32000
    ZSTD = 32015
    JPEG_LS = 32012
    BLOSC = 32001


class BloscCompressor(IntEnum):
    # https://github.com/Blosc/c-blosc/blob/master/blosc/blosc.h#L66
    BLOSCLZ = 0
    LZ4 = 1
    LZ4HC = 2
    SNAPPY = 3
    ZLIB = 4
    ZSTD = 5


# https://github.com/Blosc/hdf5-blosc/blob/master/src/example.c#L87
def get_blosc_opts(*,
                   level=4,
                   shuffle=False,
                   compressor=BloscCompressor.BLOSCLZ):
    return (0, 0, 0, 0,
            int(level), int(shuffle), int(compressor))
