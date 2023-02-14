#!/usr/bin/env python

import argparse
import cv2
import logging as log
import numpy as np
import scipy as sp

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

if __name__ == "__main__":
    args = parser.parse_args()
    log.basicConfig(level=args.log_level)

    try:
        log.info("loading image")
        image = cv2.imread(args.image)
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
    basis = np.array(
        [
            [1, 0, 0],  # C
            [0, 1, 0],  # M
            [0, 0, 1],  # Y
        ],
        dtype=np.float32,
    )

    cmy = 255 - im_arr

    log.info("writing out CMYK images")
    cv2.imwrite("c.png", (cmy[:, :, 2] > 250).astype(np.uint8) * 255)
    cv2.imwrite("m.png", (cmy[:, :, 1] > 250).astype(np.uint8) * 255)
    cv2.imwrite("y.png", (cmy[:, :, 0] > 250).astype(np.uint8) * 255)
    cv2.imwrite("k.png", (np.prod(cmy, axis=2, dtype=np.uint32) / 255**2 > 250).astype(np.uint8) * 255)
