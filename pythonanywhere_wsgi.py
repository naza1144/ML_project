"""WSGI configuration for PythonAnywhere.

Do NOT run this file locally or on `python app.py` — it is a template to copy
into PythonAnywhere's own WSGI config file (Web tab -> click the
".../wsgi.py" link near the top of the page), which PythonAnywhere loads
automatically to start your Flask app. It replaces whatever PythonAnywhere's
default file contains; keep only what's below.

Before pasting this in, change PROJECT_DIR to match where you actually
uploaded/cloned this project on PythonAnywhere (check the Files tab -- the
folder that directly contains app.py).
"""

import sys

PROJECT_DIR = "/home/naza1144/ML_project"  # <-- change this to your real path

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from app import app as application  # noqa: E402  (import after sys.path setup, on purpose)
