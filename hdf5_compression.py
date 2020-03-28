import io
import os
from os.path import join, basename
import shutil
import glob
import tempfile
import argparse
import timeit
from enum import IntEnum

import h5py
import numpy as np
import requests
import PIL.Image


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


def dl_image(ident, dtype=np.uint8):
    url = f"http://sipi.usc.edu/database/download.php?vol=textures&img={ident}"
    req = requests.get(url)
    img = PIL.Image.open(io.BytesIO(req.content))
    return np.asarray(img, dtype=dtype)


def dl_and_save_image(ident, destdir, dtype=np.uint8):
    img = dl_image(ident, dtype)
    fpath = join(destdir, f"{ident}.npy")
    np.save(fpath, img)
    return fpath


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--img-dir")
    parser.add_argument("-n", "--pixel-depth", type=int, choices=[8, 16],
                        default=8)
    parser.add_argument("-c", "--clean-img-dir", action="store_true")

    return parser.parse_args()


def benchmark_container(filter_id, filter_opts, img_dir, N=3):
    # load images
    imgs = []
    for img_path in glob.iglob(join(img_dir, "*.npy")):
        ident = basename(img_path).rstrip(".npy")
        imgs.append((ident, np.load(img_path)))

    with tempfile.TemporaryDirectory() as testdir:
        h5_path = join(testdir, "test.h5")

        # WRITE

        stmt = """
with h5py.File(h5_path, "w") as fp:
    for k in range(N):
        for ident, data in imgs:
            fp.create_dataset(f"{ident}_{k}",
                                data=data,
                                compression=filter_id,
                                compression_opts=filter_opts)
"""
        namespace = globals()
        namespace.update(locals())

        time_w = timeit.timeit(stmt, globals=namespace, number=10)
        size = os.stat(h5_path).st_size

        # READ

        ident = imgs[0][0]

        stmt = """
with h5py.File(h5_path, "r") as fp:
    img = fp[f"{ident}_0"][()]
    _ = img.shape
"""
        namespace = globals()
        namespace.update(locals())

        time_r = timeit.timeit(stmt, globals=namespace, number=100)

        return time_w, time_r, size


def main():
    args = get_args()

    if args.img_dir is None:
        img_idents = ["1.3.03", "1.3.05", "1.3.08"]

        args.img_dir = tempfile.mkdtemp()
        for ident in img_idents:
            dtype = np.uint16 if args.pixel_depth == 16 else np.uint8
            fpath = dl_and_save_image(ident, args.img_dir, dtype)
    print(f"Image source: {args.img_dir}")

    # ====================

    filters = [("none", None, None),
               ("lzf", FilterID.LZF, None),
               ("zstd", FilterID.ZSTD, None),
               ("jpegls", FilterID.JPEG_LS, None),
               ("blosc-zlib-6", FilterID.BLOSC,
                get_blosc_opts(level=6, compressor=BloscCompressor.ZLIB)),
               ("blosc-zstd-6", FilterID.BLOSC,
                get_blosc_opts(level=6, compressor=BloscCompressor.ZSTD)),
               ]

    for name, filter_id, filter_opts in filters:
        print(name, end=": ", flush=True)
        time_w, time_r, size = benchmark_container(filter_id=filter_id,
                                                   filter_opts=filter_opts,
                                                   img_dir=args.img_dir,
                                                   N=10)
        print(f"W:{time_w*1000:.0f} ms, R:{time_r*1000:.0f} ms, "
              f"{size/1e6:.2f} MB")

    # ====================

    if args.clean_img_dir:
        shutil.rmtree(args.img_dir)


if __name__ == "__main__":
    main()
