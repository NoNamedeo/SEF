Signal extraction framework for Python, for analyzing videos and sequences of images with computer vision techniques.

## Streamlit UI

The `ui/app.py` application exposes the current tracking workflow through Streamlit:

- demo video selection or local upload
- manual ROI initialization on the first frame
- OpenCV tracker execution with live preview updates
- raw vs smoothed signal charts
- CSV export of extracted signal data

Run it with:

```bash
streamlit run ui/app.py
```
