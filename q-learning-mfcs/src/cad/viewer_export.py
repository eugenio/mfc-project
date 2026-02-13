"""Export MFC stack assembly as an interactive HTML 3D viewer.

Tessellates each named part of the assembly and embeds the mesh data
into an HTML file with a three.js WebGL renderer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def build_and_export_html(output_path: str | Path) -> Path:
    """Build the full assembly and export as interactive HTML viewer."""
    import cadquery as cq
    from OCP.gp import gp_Pnt

    from .assembly import MFCStackAssembly, _COLOURS
    from .cad_config import StackCADConfig

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = StackCADConfig()

    print("Building full assembly with all subsystems...")
    builder = MFCStackAssembly(
        cfg,
        include_supports=True,
        include_labels=True,
        include_hydraulics=True,
        include_peripherals=True,
    )
    asm = builder.build()

    # Colour lookup from assembly palette
    def _get_colour(name: str) -> tuple[float, float, float]:
        for key, rgb in _COLOURS.items():
            if key in name:
                return rgb
        return (0.7, 0.7, 0.7)

    # Extract colour from CQ Color object
    def _extract_color(cq_color) -> tuple[float, float, float] | None:
        if cq_color is None:
            return None
        # cq.Color wraps Quantity_ColorRGBA
        try:
            wrapped = cq_color.wrapped
            rgb = wrapped.GetRGB()
            return (rgb.Red(), rgb.Green(), rgb.Blue())
        except Exception:
            pass
        # Maybe it IS a Quantity_ColorRGBA directly
        try:
            rgb = cq_color.GetRGB()
            return (rgb.Red(), rgb.Green(), rgb.Blue())
        except Exception:
            pass
        return None

    # Extract transform from location
    def _get_transform(loc):
        """Get gp_Trsf from loc (cq.Location or TopLoc_Location)."""
        if loc is None:
            return None
        # cq.Location has .wrapped -> TopLoc_Location -> .Transformation()
        try:
            return loc.wrapped.Transformation()
        except Exception:
            pass
        # Raw TopLoc_Location -> .Transformation()
        try:
            return loc.Transformation()
        except Exception:
            pass
        return None

    parts_data = []

    for name, child in asm.objects.items():
        if child.obj is None:
            continue

        try:
            shape = child.obj
            if hasattr(shape, 'val'):
                solid = shape.val()
            else:
                solid = shape

            verts, faces = solid.tessellate(0.5)

            if not verts:
                continue

            # Transform vertices by location
            trsf = _get_transform(child.loc)
            vertices = []
            if trsf is not None:
                for v in verts:
                    pt = gp_Pnt(v.x, v.y, v.z)
                    pt_t = pt.Transformed(trsf)
                    vertices.extend([pt_t.X(), pt_t.Y(), pt_t.Z()])
            else:
                for v in verts:
                    vertices.extend([v.x, v.y, v.z])

            indices = []
            for f in faces:
                indices.extend(f)

            # Determine colour
            color = _extract_color(child.color)
            if color is None:
                color = _get_colour(str(name))
            r, g, b = color

            parts_data.append({
                "name": str(name),
                "vertices": vertices,
                "indices": indices,
                "color": [r, g, b],
            })
        except Exception as e:
            print(f"  Skipping {name}: {e}")
            continue

    print(f"Collected {len(parts_data)} parts with mesh data")

    html = _generate_html(parts_data, cfg)
    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote viewer to {output_path}")
    return output_path


def _generate_html(parts_data: list[dict], cfg) -> str:
    """Generate a standalone HTML file with three.js 3D viewer."""

    parts_json = json.dumps(parts_data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MFC Stack — {cfg.num_cells}-Cell Assembly</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #1a1a2e; overflow: hidden; font-family: monospace; }}
  canvas {{ display: block; }}
  #info {{
    position: absolute; top: 10px; left: 10px;
    color: #e0e0e0; background: rgba(0,0,0,0.6);
    padding: 12px 16px; border-radius: 6px;
    font-size: 13px; line-height: 1.6;
    pointer-events: none; z-index: 10;
  }}
  #info h2 {{ margin-bottom: 6px; color: #4fc3f7; font-size: 15px; }}
  #controls {{
    position: absolute; bottom: 10px; left: 10px;
    color: #aaa; font-size: 11px;
    background: rgba(0,0,0,0.4); padding: 8px 12px; border-radius: 4px;
    z-index: 10;
  }}
</style>
</head>
<body>
<div id="info">
  <h2>MFC Stack — {cfg.num_cells}-Cell Assembly</h2>
  Stack length: {cfg.stack_length * 1000:.0f} mm<br>
  Outer frame: {cfg.outer_side * 1000:.0f} x {cfg.outer_side * 1000:.0f} mm<br>
  Cell thickness: {cfg.cell_thickness * 1000:.0f} mm<br>
  Parts in view: <span id="partCount">0</span>
</div>
<div id="controls">
  Drag to rotate | Scroll to zoom | Right-drag to pan
</div>

<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.168.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.168.0/examples/jsm/"
  }}
}}
</script>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);
scene.fog = new THREE.Fog(0x1a1a2e, 2000, 4000);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 10000);

const renderer = new THREE.WebGLRenderer({{ antialias: true, powerPreference: "high-performance" }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
document.body.appendChild(renderer.domElement);

// Lighting
scene.add(new THREE.AmbientLight(0x404060, 0.8));

const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
dirLight.position.set(300, 500, 400);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(2048, 2048);
scene.add(dirLight);

const fillLight = new THREE.DirectionalLight(0x4488cc, 0.6);
fillLight.position.set(-200, 100, -300);
scene.add(fillLight);

const rimLight = new THREE.DirectionalLight(0xffaa44, 0.3);
rimLight.position.set(0, -200, 200);
scene.add(rimLight);

// Ground + grid
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(4000, 4000),
  new THREE.MeshStandardMaterial({{ color: 0x222244, roughness: 0.9 }})
);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -120;
ground.receiveShadow = true;
scene.add(ground);

const grid = new THREE.GridHelper(3000, 60, 0x444466, 0x333355);
grid.position.y = -119;
scene.add(grid);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

// Load mesh parts
const partsData = {parts_json};
let partCount = 0;
const group = new THREE.Group();

for (const part of partsData) {{
  if (!part.vertices.length) continue;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(part.vertices), 3));
  if (part.indices.length) geo.setIndex(part.indices);
  geo.computeVertexNormals();

  const [r, g, b] = part.color;
  const mat = new THREE.MeshStandardMaterial({{
    color: new THREE.Color(r, g, b),
    roughness: 0.4, metalness: 0.15,
    side: THREE.DoubleSide,
  }});
  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.name = part.name;
  group.add(mesh);
  partCount++;
}}

scene.add(group);
document.getElementById('partCount').textContent = partCount;

// Frame the model
const box = new THREE.Box3().setFromObject(group);
const center = box.getCenter(new THREE.Vector3());
const size = box.getSize(new THREE.Vector3());
const maxDim = Math.max(size.x, size.y, size.z);

controls.target.copy(center);
camera.position.set(center.x + maxDim * 0.9, center.y + maxDim * 0.5, center.z + maxDim * 0.9);
controls.update();

(function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}})();

window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});
</script>
</body>
</html>"""


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    out = sys.argv[1] if len(sys.argv) > 1 else "mfc_stack_viewer.html"
    build_and_export_html(out)
