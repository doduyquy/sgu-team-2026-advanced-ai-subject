// Install build-time dependencies in artifacts/vehicle-asset-tools (see credits).
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
const require = createRequire(new URL('../artifacts/vehicle-asset-tools/package.json', import.meta.url));
const load = async name => import(pathToFileURL(require.resolve(name)).href);
const { NodeIO } = await load('@gltf-transform/core');
const { ALL_EXTENSIONS } = await load('@gltf-transform/extensions');
const { dedup, weld, simplify, prune } = await load('@gltf-transform/functions');
const { MeshoptDecoder, MeshoptSimplifier } = await load('meshoptimizer');
const draco = require('draco3dgltf');
await Promise.all([MeshoptDecoder.ready, MeshoptSimplifier.ready]);
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({
  'meshopt.decoder': MeshoptDecoder,
  'draco3d.decoder': await draco.createDecoderModule(),
});
const [input, output, ratio = '1'] = process.argv.slice(2);
if (!input || !output) throw new Error('Usage: node scripts/optimize_vehicle_assets.mjs input.glb output.glb [ratio]');
const document = await io.read(resolve(input));
for (const extension of document.getRoot().listExtensionsUsed()) {
  if (['EXT_meshopt_compression','KHR_draco_mesh_compression'].includes(extension.extensionName)) extension.dispose();
}
await document.transform(dedup(), weld(), simplify({simplifier:MeshoptSimplifier,ratio:Number(ratio),error:.002}), prune());
await io.write(resolve(output),document);
console.log('Wrote standard glTF without runtime decoder:',output);
