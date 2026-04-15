"""
Compatibility shim: restores `image_to_url` in `streamlit.elements.image`.

`streamlit-drawable-canvas` 0.9.3 was written against an older Streamlit
internal API that exposed `image_to_url` in `streamlit.elements.image`.
That function was removed in Streamlit ≥ 1.20.  The canvas package has not
been updated accordingly.

This module patches the missing function **before** `streamlit_drawable_canvas`
is imported, so it must be imported first in any module that uses `st_canvas`.

The patch converts PIL Images and numpy arrays to inline base-64 data-URLs,
which the canvas React component accepts as a valid `<img src>` value.
"""
from __future__ import annotations

import base64
import io

import streamlit.elements.image as _st_image

if not hasattr(_st_image, "image_to_url"):
    import numpy as np
    from PIL import Image as _PIL

    def _image_to_url(
        image,
        width: int = -1,
        clamp: bool = False,
        channels: str = "RGB",
        output_format: str = "auto",
        image_id: str = "",
    ) -> str:
        """Return a base-64 data-URL for *image* (PIL Image or numpy array)."""
        if isinstance(image, _PIL.Image):
            pil = image
        elif isinstance(image, np.ndarray):
            pil = _PIL.fromarray(image.astype("uint8"))
        else:
            return ""

        fmt = (
            "PNG"
            if output_format.lower() in ("auto", "")
            else output_format.upper()
        )
        buf = io.BytesIO()
        pil.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/{fmt.lower()};base64,{b64}"

    _st_image.image_to_url = _image_to_url
