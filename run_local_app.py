import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from retail_theft_detector.web_app import main  # noqa: E402

if __name__ == "__main__":
    main()
