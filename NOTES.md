## oddities, subtleties, limitations, bugs, TODOs

- insufficient lighting makes the framerate drop (because of auto-exposure)
    - turn off auto-exposure, warn user when images are too dark?
    - should choose max. exposure that still captures an image every 1/60 secs
- face detection range was *very* limited (50cm or so on my setup) before I started cropping the input image
    - was caused by aspect-aware image resizing
    - currently using short-range model, should try the long range sparse variant
      (once https://github.com/onnx/tensorflow-onnx/issues/1877 is fixed)
    - long range non-sparse variant needs `Upsample` op, which tract doesn't implement yet
