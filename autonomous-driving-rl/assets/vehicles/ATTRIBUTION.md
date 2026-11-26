# Vehicle artwork attribution

These community meshes depict production vehicles. Their render-only use does
not change MetaDrive physics or represent manufacturer-calibrated dynamics.

## Lamborghini Aventador

- Artist: [Arion Digital](https://sketchfab.com/andrewswihart).
- Original: [Lamborghini Aventador](https://sketchfab.com/3d-models/lamborghini-aventador-888e37a3641d4f7b94bc1a39396e2441).
- License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
- Downloaded public copy: [immx2/showcase-01](https://github.com/immx2/showcase-01/blob/main/public/models/lamborghini.glb).
- Modifications: runtime scale/orientation, separate wheel pivots, copied and
  resized/compressed artwork textures, PBR environment and display adaptation.
- Road stance: body height aligned to the loaded MetaDrive suspension, wheel
  radii fitted to Bullet and original lateral hub positions retained visually.
  All four imported wheels inherit physical rolling and front-wheel steering.
  The isolated Garage preview retains the source model's static hub positions.
- Original complete mesh: 10,252 triangles; body 7,396, four wheels 2,856.

## Tesla Cybertruck (2025)

- Artist/uploader: Nieve5677.
- Original: [Tesla Cybertruck 2025](https://sketchfab.com/3d-models/tesla-cybertruck-2025-0fe4980c8cbc441382bfb7d4cf9f092e).
- License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
- Credited public copy: [Tesla Studio credits](https://github.com/aditano/tesla-studio/blob/main/public/models/cybertruck-import/CREDITS.md).
- Modifications: decoded compression, removed three detached floating blocks,
  corrected width/orientation, separated four wheel assemblies, assigned six
  materials, planarized side panels and rebuilt flat body normals. Meshopt
  simplification at ratio 0.35 produces 42,161 triangles from 72,528 retained
  source triangles. The final GLB is 2,074,320 bytes and needs no textures or
  runtime decoder. LED strips and dark side glazing are authored runtime meshes.
- All four imported wheels inherit Bullet rolling and front-wheel steering.
  Only the artwork is fitted to the common simulator chassis; physical mass,
  wheel pivots and race limits are shared with the other vehicle appearances.
  These dimensions and dynamics are not manufacturer calibration.

## VinFast VF3

The VF3 is an original illustrative mesh authored in `vf3_mesh.py`, with
wheel artwork in `race_car_mesh.py`. It was revised to follow the user's
turquoise body/black roof photo reference: upright two-door body, black roof,
wide arch cladding, rectangular headlights, V-shaped trim and black steel
wheels. It is not a downloaded manufacturer model or calibrated CAD.
No paid VF3 asset was purchased.

Rebuild Tesla using `scripts/prepare_cybertruck_asset.py` followed by
`scripts/optimize_vehicle_assets.mjs`; commands are in `docs/VEHICLE_SHOWROOM.md`.
The optimizer's build-time npm
dependencies are isolated in `artifacts/vehicle-asset-tools`. Source SHA-256
and final-file hashes are recorded in `manifest.json` beside this notice.
