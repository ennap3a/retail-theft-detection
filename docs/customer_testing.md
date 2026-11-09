# Customer Testing Workflow

This workflow is for testing the product before training a custom model.

You do not need to create classes, labels, or datasets for this step. You only act like a customer.

## What You Need

Prepare two folders on your computer:

```text
test_videos/
  normal/
  shoplifting/
```

Put normal shopping/browsing videos in `normal`.

Put acted or real shoplifting/suspicious concealment videos in `shoplifting`.

## Test Steps

1. Run the app:

```bash
python run_local_app.py
```

2. Open:

```text
http://127.0.0.1:7860
```

3. Upload one video. Do not tell the app whether it is normal or shoplifting.

4. Draw zones:

- Product zone: shelves/tables/racks where items are taken.
- Customer zone: browsing/standing area.

5. Save setup.

6. Open project summary and click **Run detector**.

7. Open results.

The results page shows:

```text
number of human-review alerts
annotated review video
alert evidence if any
summary CSV
```

You should keep the real answer in your own testing notes, outside the app.

## How To Interpret

For a normal video:

- `0 alerts` is good.
- alerts mean possible false alarm.

For a shoplifting/suspicious video:

- `1+ alerts` is good.
- `0 alerts` means possible missed event.

## What To Send Back

After testing several videos, send:

- how many normal videos passed
- how many normal videos caused false alerts
- how many shoplifting videos passed
- how many shoplifting videos were missed
- screenshots or exported project results if something failed

This tells us whether we need to improve rules, train a custom model, or collect better labels.
