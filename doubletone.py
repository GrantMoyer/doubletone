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
    default=None,
)
parser.add_argument(
    "-M",
    "--magenta-out",
    help="Color of magenta in output image",
    type=hex_color,
    default=None,
)
parser.add_argument(
    "-Y",
    "--yellow-out",
    help="Color of yellow in output image",
    type=hex_color,
    default=None,
)
parser.add_argument(
    "-K",
    "--black-out",
    help="Color of black in output image",
    type=hex_color,
    default=None,
)
parser.add_argument(
    "--cyan-angle",
    help="Angle of filter for cyan in turns",
    type=float,
    default=3 / 16,
)
parser.add_argument(
    "--magenta-angle",
    help="Angle of filter for magenta in turns",
    type=float,
    default=2 / 16,
)
parser.add_argument(
    "--yellow-angle",
    help="Angle of filter for yellow in turns",
    type=float,
    default=0,
)
parser.add_argument(
    "--black-angle",
    help="Angle of filter for black in turns",
    type=float,
    default=1 / 16,
)
parser.add_argument(
    "-t",
    "--black-threshold",
    help="Threshold to consider pixel black",
    type=float,
    default=0.1,
)
parser.add_argument(
    "-s",
    "--filter-window",
    help="Window width of low pass filter in pixels",
    type=float,
    default=12,
)


def intensity_from_srgb(image):
    gamma = 2.4
    A = 0.055
    phi = 12.92
    X = 0.04045
    image = image.astype(np.float32)
    image /= 255.0
    linear_region = image < X
    image[linear_region] /= phi
    image_non_linear_region = image[np.logical_not(linear_region)]
    del linear_region
    image_non_linear_region += A
    image_non_linear_region /= 1.0 + A
    image_non_linear_region **= gamma
    return image


def srgb_from_intensity(intensity):
    gamma = 2.4
    A = 0.055
    phi = 12.92
    X = 0.04045
    linear_region = intensity < X / phi
    intensity[linear_region] *= phi
    intensity_non_linear_region = intensity[np.logical_not(linear_region)]
    del linear_region
    intensity_non_linear_region **= 1.0 / gamma
    intensity_non_linear_region *= 1.0 + A
    intensity_non_linear_region -= A

    intensity *= 255.0
    intensity.round(out=intensity)
    intensity.clip(0.0, 255.0, out=intensity)
    return intensity.astype(np.uint8)


def cmy_from_bgr(bgr_intensity, cyan, magenta, yellow):
    cmy_srgb = np.array(
        [
            cyan,
            magenta,
            yellow,
        ]
    )
    cmy_intensity = intensity_from_srgb(cmy_srgb)
    white_intensity = np.array([1.0, 1.0, 1.0])
    basis = white_intensity - cmy_intensity

    cmy = white_intensity - bgr_intensity
    cmy = cmy @ np.linalg.inv(basis)
    return cmy


def bgr_from_cmy(cmy, cyan, magenta, yellow):
    cmy_srgb = np.array(
        [
            cyan,
            magenta,
            yellow,
        ]
    )
    cmy_intensity = intensity_from_srgb(cmy_srgb)
    white_intensity = np.array([1.0, 1.0, 1.0])
    basis = white_intensity - cmy_intensity

    bgr_intensity = cmy @ basis
    bgr_intensity = white_intensity - bgr_intensity
    return bgr_intensity


def lanczos(scale, lobes=3):
    bound = int(np.round(scale * lobes))
    samples = bound * 2 - 1
    x = np.linspace(-bound / scale, bound / scale, samples)
    L = np.sinc(x) * np.sinc(x / lobes)
    return L / L.sum()


if __name__ == "__main__":
    args = parser.parse_args()
    log.basicConfig(level=args.log_level)
    for color in "cyan", "magenta", "yellow", "black":
        if vars(args)[f"{color}_out"] is None:
            vars(args)[f"{color}_out"] = vars(args)[f"{color}_in"]

    try:
        log.info("loading image")
        image = cv2.imread(args.image, flags=cv2.IMREAD_COLOR)
    except Exception as e:
        log.critical(f"Failed to open image: {e}")
        exit(1)

    log.info("converting to CMYK")
    bgr_intensity = intensity_from_srgb(image)
    del image
    cmy = cmy_from_bgr(bgr_intensity, args.cyan_in, args.magenta_in, args.yellow_in)
    black_intensity = intensity_from_srgb(args.black_in)
    k = (
        np.linalg.norm(bgr_intensity - black_intensity, axis=2) < args.black_threshold
    ).astype(np.float32)
    del bgr_intensity
    width, height, channels = cmy.shape

    log.info("descreening image")
    c = sp.ndimage.rotate(cmy[:, :, 0], -360 * args.cyan_angle, prefilter=False)
    m = sp.ndimage.rotate(cmy[:, :, 1], -360 * args.magenta_angle, prefilter=False)
    y = sp.ndimage.rotate(cmy[:, :, 2], -360 * args.yellow_angle, prefilter=False)
    k = sp.ndimage.rotate(k, -360 * args.black_angle, prefilter=False)
    del cmy

    kernel = lanczos(args.filter_window)

    log.debug("descreening cyan channel")
    c = sp.ndimage.convolve1d(c, kernel, axis=0)
    c = sp.ndimage.convolve1d(c, kernel, axis=1)
    c = sp.ndimage.rotate(c, 360 * args.cyan_angle, prefilter=False, reshape=False)
    c_width, c_height = c.shape
    border_width = (c_width - width) // 2
    border_height = (c_height - height) // 2
    c = c[border_width : border_width + width, border_height : border_height + height]
    assert c.shape == (width, height)

    log.debug("descreening magenta channel")
    m = sp.ndimage.convolve1d(m, kernel, axis=0)
    m = sp.ndimage.convolve1d(m, kernel, axis=1)
    m = sp.ndimage.rotate(m, 360 * args.magenta_angle, prefilter=False, reshape=False)
    m_width, m_height = m.shape
    border_width = (m_width - width) // 2
    border_height = (m_height - height) // 2
    m = m[border_width : border_width + width, border_height : border_height + height]
    assert m.shape == (width, height)

    log.debug("descreening yellow channel")
    y = sp.ndimage.convolve1d(y, kernel, axis=0)
    y = sp.ndimage.convolve1d(y, kernel, axis=1)
    y = sp.ndimage.rotate(y, 360 * args.yellow_angle, prefilter=False, reshape=False)
    y_width, y_height = y.shape
    border_width = (y_width - width) // 2
    border_height = (y_height - height) // 2
    y = y[border_width : border_width + width, border_height : border_height + height]
    assert y.shape == (width, height)

    log.debug("descreening black channel")
    k = sp.ndimage.convolve1d(k, kernel, axis=0)
    k = sp.ndimage.convolve1d(k, kernel, axis=1)
    k = sp.ndimage.rotate(k, 360 * args.black_angle, prefilter=False, reshape=False)
    k_width, k_height = k.shape
    border_width = (k_width - width) // 2
    border_height = (k_height - height) // 2
    k = k[border_width : border_width + width, border_height : border_height + height]
    k **= 2.0
    assert k.shape == (width, height)

    filtered = np.stack([c, m, y], axis=2)
    del c
    del m
    del y

    log.info("re-combining CMYK")
    combined_intensity = bgr_from_cmy(
        filtered, args.cyan_out, args.magenta_out, args.yellow_out
    )
    black_out_intensity = intensity_from_srgb(args.black_out)
    combined_intensity = np.expand_dims(
        1 - k, axis=2
    ) * combined_intensity + np.expand_dims(k, axis=2) * np.expand_dims(
        black_out_intensity, axis=(0, 1)
    )
    del k

    combined = srgb_from_intensity(combined_intensity)
    del combined_intensity

    log.info("writing out filtered image")
    cv2.imwrite("filtered.png", combined)
