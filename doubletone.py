#!/usr/bin/env python

import argparse
import cv2
import logging as log
import numpy as np
import re
import scipy as sp


def hex_color(hex):
    m = re.fullmatch(r"#?([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})", hex)
    if not m:
        raise ValueError("Failed to parse hex color: {hex}")
    return np.array([int(m[i], base=16) for i in [3, 2, 1]], dtype=np.uint8)


parser = argparse.ArgumentParser(
    description="Filters a halftone pattern to produce an image more suitable for digital displays"
)
parser.add_argument("image", help="Image to filter")
parser.add_argument(
    "-l",
    "--log-level",
    help="Verbosity of logging",
    choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"),
    type=str.upper,
    default=log.NOTSET,
)
parser.add_argument(
    "-t",
    "--threshold",
    help="Threshold for color to be considered like C, M, or Y",
    type=float,
    default=0.9,
)
parser.add_argument(
    "-c",
    "--cyan-in",
    help="Color of cyan in input image",
    type=hex_color,
    default="#00ffff",
)
parser.add_argument(
    "-m",
    "--magenta-in",
    help="Color of magenta in input image",
    type=hex_color,
    default="#ff00ff",
)
parser.add_argument(
    "-y",
    "--yellow-in",
    help="Color of yellow in input image",
    type=hex_color,
    default="#ffff00",
)
parser.add_argument(
    "-k",
    "--black-in",
    help="Color of black in input image",
    type=hex_color,
    default="#000000",
)
parser.add_argument(
    "-C",
    "--cyan-out",
    help="Color of cyan in output image",
    type=hex_color,
    default="#00ffff",
)
parser.add_argument(
    "-M",
    "--magenta-out",
    help="Color of magenta in output image",
    type=hex_color,
    default="#ff00ff",
)
parser.add_argument(
    "-K",
    "--black-out",
    help="Color of black in output image",
    type=hex_color,
    default="#000000",
)

if __name__ == "__main__":
    args = parser.parse_args()
    log.basicConfig(level=args.log_level)

    try:
        log.info("loading image")
        image = cv2.imread(args.image, flags=cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
        image = image.astype(np.float32) / 255.0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
        im_arr = np.asarray(image)
    except Exception as e:
        log.critical(f"Failed to open image: {e}")
        exit(1)

    # log.info("computing ink colors")
    # log.debug("selecting random subset of pixels")
    # pixels = im_arr.reshape(-1, im_arr.shape[-1])
    # SUBSET_MAX_SIZE = 1000000
    # subset = np.random.default_rng().choice(
    #    pixels, size=min(SUBSET_MAX_SIZE, pixels.shape[0]), replace=False, shuffle=False
    # )

    # log.debug("running k-means clustering on subset")
    # CMYK_COLORS = np.array(
    #    [
    #        [50, 150, 150],  # C
    #        [150, 50, 150],  # M
    #        [150, 150, 50],  # Y
    #        [0, 0, 100],  # C + M
    #        [100, 0, 0],  # M + Y
    #        [0, 100, 0],  # Y + C
    #        [0, 0, 0],  # K
    #        [255, 255, 255],  # white, no ink
    #    ]
    # )
    # colors, labels = sp.cluster.vq.kmeans2(subset.astype(np.float32), 20, minit="++", missing="warn")

    log.info("converting to CMYK")
    cmy_sRGB = (
        np.array(
            [
                args.cyan_in,
                args.magenta_in,
                args.yellow_in,
            ],
            dtype=np.float32,
        )
        / 255.0
    )
    cmy_xyz = cv2.cvtColor(np.expand_dims(cmy_sRGB, axis=0), cv2.COLOR_BGR2XYZ)[0]
    white_xyz = cv2.cvtColor(
        np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32), cv2.COLOR_BGR2XYZ
    )[0, 0]
    basis = white_xyz - cmy_xyz

    cmy = (white_xyz - im_arr) @ np.linalg.inv(basis)

    cyan = cmy[:, :, 0]
    magenta = cmy[:, :, 1]
    yellow = cmy[:, :, 2]
    black = np.prod(cmy, axis=2, dtype=np.uint32) / 255**2

    log.info("re-combining CMYK")
    filtered = np.full(im_arr.shape, 255, dtype=np.uint8)
    filtered[cyan > args.threshold, 2] = 0
    filtered[magenta > args.threshold, 1] = 0
    filtered[yellow > args.threshold, 0] = 0

    log.info("writing out filterd image")
    cv2.imwrite("filtered.png", filtered)
