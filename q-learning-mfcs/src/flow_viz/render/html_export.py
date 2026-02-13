"""Interactive HTML viewer for flow visualization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from flow_viz.config import FlowVizConfig


def export_flow_html(
    streamlines: list[dict[str, Any]],
    viz_config: FlowVizConfig | None = None,
    output_path: Path | str | None = None,
    title: str = "MFC Flow Visualization",
) -> Path:
    """Export streamlines to an interactive HTML viewer."""
    if viz_config is None:
        viz_config = FlowVizConfig()
    if output_path is None:
        output_path = viz_config.output_path("flow_viewer")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines_data = _prepare_streamline_data(streamlines, viz_config)
    html = _generate_html(lines_data, title, viz_config)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _prepare_streamline_data(
    streamlines: list[dict[str, Any]],
    viz_config: FlowVizConfig,  # noqa: ARG001
) -> list[dict]:
    """Convert streamlines to JSON-serializable format."""
    result = []
    for sl in streamlines:
        pts = np.asarray(sl["points"])
        vels = np.asarray(sl["velocities"])
        v_min, v_max = vels.min(), vels.max()
        if v_max > v_min:
            v_norm = ((vels - v_min) / (v_max - v_min)).tolist()
        else:
            v_norm = [0.5] * len(vels)
        result.append({"points": pts.tolist(), "colors": v_norm})
    return result


def _generate_html(
    lines_data: list[dict],
    title: str,
    viz_config: FlowVizConfig,
) -> str:
    """Generate standalone HTML with three.js flow viewer."""
    data_json = json.dumps(lines_data)
    bg = viz_config.background_color
    return _html_template(data_json, title, bg)


def _html_template(data_json: str, title: str, bg: str) -> str:
    """Build the HTML string."""
    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        f"<title>{title}</title>\n"
        "<style>\n"
        "* { margin: 0; padding: 0; box-sizing: border-box; }\n"
        f"body {{ background: {bg}; overflow: hidden; "
        "font-family: monospace; }}\n"
        "canvas { display: block; }\n"
        "#info { position: absolute; top: 10px; left: 10px;\n"
        "  color: #e0e0e0; background: rgba(0,0,0,0.6);\n"
        "  padding: 12px 16px; border-radius: 6px;\n"
        "  font-size: 13px; line-height: 1.6;\n"
        "  pointer-events: none; z-index: 10; }\n"
        "#info h2 { margin-bottom: 6px; color: #4fc3f7;"
        " font-size: 15px; }\n"
        "#controls { position: absolute; bottom: 10px; left: 10px;\n"
        "  color: #aaa; font-size: 11px;\n"
        "  background: rgba(0,0,0,0.4); padding: 8px 12px;\n"
        "  border-radius: 4px; z-index: 10; }\n"
        "</style>\n</head>\n<body>\n"
        '<div id="info">\n'
        f"  <h2>{title}</h2>\n"
        '  Streamlines: <span id="lineCount">0</span>\n'
        "</div>\n"
        '<div id="controls">\n'
        "  Drag to rotate | Scroll to zoom | Right-drag to pan\n"
        "</div>\n"
        '<script type="importmap">\n'
        "{\n"
        '  "imports": {\n'
        '    "three": '
        '"https://cdn.jsdelivr.net/npm/three@0.168.0'
        '/build/three.module.js",\n'
        '    "three/addons/": '
        '"https://cdn.jsdelivr.net/npm/three@0.168.0'
        '/examples/jsm/"\n'
        "  }\n}\n"
        "</script>\n"
        '<script type="module">\n'
        "import * as THREE from 'three';\n"
        "import { OrbitControls } from "
        "'three/addons/controls/OrbitControls.js';\n"
        "const scene = new THREE.Scene();\n"
        f"scene.background = new THREE.Color('{bg}');\n"
        "const camera = new THREE.PerspectiveCamera("
        "45, window.innerWidth / window.innerHeight, 0.1, 10000);\n"
        "const renderer = new THREE.WebGLRenderer({ antialias: true });\n"
        "renderer.setSize(window.innerWidth, window.innerHeight);\n"
        "renderer.setPixelRatio(window.devicePixelRatio);\n"
        "document.body.appendChild(renderer.domElement);\n"
        "scene.add(new THREE.AmbientLight(0x404060, 0.8));\n"
        "const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);\n"
        "dirLight.position.set(300, 500, 400);\n"
        "scene.add(dirLight);\n"
        "const controls = new OrbitControls("
        "camera, renderer.domElement);\n"
        "controls.enableDamping = true;\n"
        f"const linesData = {data_json};\n"
        "let lineCount = 0;\n"
        "const group = new THREE.Group();\n"
        "for (const line of linesData) {\n"
        "  const pts = line.points;\n"
        "  if (pts.length < 2) continue;\n"
        "  const positions = [];\n"
        "  const colors = [];\n"
        "  for (let i = 0; i < pts.length; i++) {\n"
        "    positions.push(pts[i][0], pts[i][1], pts[i][2]);\n"
        "    const t = (line.colors && line.colors[i]) || 0.5;\n"
        "    colors.push(t, 1.0 - t, 0.5);\n"
        "  }\n"
        "  const geo = new THREE.BufferGeometry();\n"
        "  geo.setAttribute('position', "
        "new THREE.Float32BufferAttribute(positions, 3));\n"
        "  geo.setAttribute('color', "
        "new THREE.Float32BufferAttribute(colors, 3));\n"
        "  const mat = new THREE.LineBasicMaterial("
        "{ vertexColors: true });\n"
        "  group.add(new THREE.Line(geo, mat));\n"
        "  lineCount++;\n"
        "}\n"
        "scene.add(group);\n"
        "document.getElementById('lineCount')"
        ".textContent = lineCount;\n"
        "camera.position.set(200, 200, 200);\n"
        "controls.update();\n"
        "(function animate() {\n"
        "  requestAnimationFrame(animate);\n"
        "  controls.update();\n"
        "  renderer.render(scene, camera);\n"
        "})();\n"
        "window.addEventListener('resize', () => {\n"
        "  camera.aspect = window.innerWidth / window.innerHeight;\n"
        "  camera.updateProjectionMatrix();\n"
        "  renderer.setSize(window.innerWidth, window.innerHeight);\n"
        "});\n"
        "</script>\n</body>\n</html>"
    )
