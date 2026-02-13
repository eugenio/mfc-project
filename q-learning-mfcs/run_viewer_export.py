"""Script to generate the MFC stack 3D HTML viewer."""
import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from cad.viewer_export import build_and_export_html
    out = build_and_export_html(os.path.join(os.path.dirname(__file__), 'mfc_stack_viewer.html'))
    print(f"SUCCESS: Generated {out}")
except Exception as e:
    traceback.print_exc()
    print(f"FAILED: {e}")
    sys.exit(1)
