import base64

import cv2
import numpy as np


def read_by_b64(fpath: str) -> str:
    with open(fpath, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf8')


def b64_to_cv(im_b64: str) -> np.ndarray:
    im_bytes = base64.b64decode(im_b64)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
    return cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
