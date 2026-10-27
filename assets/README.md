# assets/

Place optional static files here (logos, images for the report or slides).

The app does not need anything from this folder to run: all styling is defined in
`utils/ui.py` (CSS + HTML components) and `.streamlit/config.toml` (theme colours).
Fonts (Barlow / Barlow Condensed) load from Google Fonts when online; without internet the
app falls back to system fonts automatically. Recommendations never need internet access.
