"""Offline material build: preserve Poly Haven's separately supplied leaf alpha.

The upstream JPG glTF names its leaf image as diffuse+alpha, but a JPG cannot
contain alpha. Pack the bundled CC0 mask before running GLB optimization.
No network request occurs at game startup.
"""
import json
from pathlib import Path
from PIL import Image


def main():
    root=Path(__file__).resolve().parents[1]/'assets/scenery/island_tree_02'
    with Image.open(root/'textures/island_tree_02_leaves_diff_1k.jpg') as diffuse:
        rgba=diffuse.convert('RGBA')
    with Image.open(root/'textures/island_tree_02_leaves_alpha_1k.png') as mask:
        rgba.putalpha(mask.convert('L'))
    rgba.save(root/'textures/island_tree_02_leaves_rgba_1k.png')
    document=json.loads((root/'island_tree_02_1k.gltf').read_text(encoding='utf-8'))
    for image in document['images']:
        if 'leaves_diff' in image['name']:
            image.update(uri='textures/island_tree_02_leaves_rgba_1k.png',mimeType='image/png')
    (root/'tree-alpha-fixed.gltf').write_text(json.dumps(document),encoding='utf-8')


if __name__=='__main__':
    main()
