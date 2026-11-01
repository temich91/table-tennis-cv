"""Точка входа для запуска дашборда Streamlit.

Использование::

    streamlit run app.py
"""

from pathlib import Path
import runpy

app_path = Path(__file__).parent / "src" / "ttcv" / "app.py"
runpy.run_path(str(app_path), run_name="__main__")
