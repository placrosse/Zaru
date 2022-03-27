# Mizaru sees no evil

## oddities, subtleties, limitations, bugs, TODOs

- insufficient lighting makes the framerate drop (because of auto-exposure)
    - turn off auto-exposure, warn user when images are too dark?
    - should choose max. exposure that still captures an image every 1/60 secs
- face detection range is *very* limited (50cm or so on my setup)
    - in part, caused by aspect-aware image resizing
    - currently using short-range model, should try the long range sparse variant
      (once https://github.com/onnx/tensorflow-onnx/issues/1877 is fixed)
- ext. resources like neural nets are loaded from disk at runtime
    - should be `include!`d instead

## other notes

- consider using hysteresis to switch between "no tracking data" and "yes tracking data"
    - this avoids spurious tracking data from incorrect face detections (false positives)
    - also avoids spurious loss of tracking from false negatives
