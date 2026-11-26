# Vietnam scenery assets

Bundled for offline use. Sources, licenses and source MD5 checksums are listed
in `sources.json`. All photographic scenery assets here are Poly Haven CC0.
The OSM route has its own ODbL attribution in `../maps/vietnam/`.
Authored streetscape placement is illustrative, not a surveyed reconstruction.

Runtime uses `tree-near-v2.glb`, `tree-far.glb`, four `facade-*.glb` units,
the brick paving diffuse and the grey plaster diffuse. Meshes and textures are
shared between instances; nearby trees, distant trees and facades have LOD.

The upstream tree's JPEG leaf material lacks the separate alpha mask. Rebuild
the packed PNG before optimizing, otherwise leaves appear as black rectangles:

```powershell
..\.venv\metadrive\python.exe scripts/pack_vietnam_tree_alpha.py
node scripts/optimize_scenery_assets.mjs assets/scenery/island_tree_02/tree-alpha-fixed.gltf assets/scenery/island_tree_02/tree-near-v2.glb .025
node scripts/optimize_scenery_assets.mjs assets/scenery/island_tree_02/tree-alpha-fixed.gltf assets/scenery/island_tree_02/tree-far.glb .006
node scripts/extract_scenery_facades.mjs
```

The optimization tools reuse the Node dependencies in
`artifacts/vehicle-asset-tools/`; dependencies are not required at game startup.
