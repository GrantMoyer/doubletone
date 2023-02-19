Doubletone removes the halftone pattern from a scan of an image printed with offset halftone.

The algorithm is roughly:

- convert he image from RGB to CMYK
- for each of the CMYK channels
  - rotate the channel to align the channel's halftone grid horizontally and vertically
  - apply a low pass filter (lanczos) to the channel horizontally and vertically
  - rotate the channel back to it's initial orientation
- re-combine the CMYK channels
- convert back from CMYK to RGB
