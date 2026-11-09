# Stage 1 Product Plan

## Goal

Replace the old four-camera Colab setup with a clean single-video workflow that can become a customer demo.

## Free First

The first version runs locally and uses only Python standard-library web serving plus existing OpenCV/Numpy/Ultralytics dependencies. This avoids paid hosting while the product is still being validated.

## Professional User Flow

1. Customer uploads one video.
2. The system extracts a setup frame.
3. Customer can choose a better setup-frame timestamp.
4. Customer draws product zones, plus optional checkout and exit zones.
5. The detector uses that setup.
6. Results become a review dashboard, not an accusation.

## Zone Guidance

Product zones should be tight around shelves, racks, counters, tables, or other item display areas. Everything outside the product zones is treated as shop floor. Checkout zones should cover tills or self-checkouts and exit zones the doors (or frame edge) people leave through; both are optional and only change alert priority.

## Accuracy Plan

Accuracy should be measured after the single-video detector is connected. The minimum dataset needs:

- true suspicious events
- normal holding events
- returned-item events
- false-alarm examples
- missed-event examples

Metrics:

- event precision
- event recall
- false alerts per hour
- timestamp error
- detection mAP for people/items/bags
- tracking ID stability

## Stage 2 Status

The first single-video detector is now connected. It runs a lightweight YOLO object model, YOLO pose model, ByteTrack tracking, product-zone frame differencing, and rule-based suspicious-event logic.

This is an MVP detector, not the final accuracy version. Its job is to make the product testable end to end so the next stage can measure false positives, missed events, and timestamp quality.
