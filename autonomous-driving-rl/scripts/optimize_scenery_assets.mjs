// Offline build step. Dependencies are the existing vehicle-asset-tools install.
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
const require=createRequire(new URL('../artifacts/vehicle-asset-tools/package.json',import.meta.url));
const load=async name=>import(pathToFileURL(require.resolve(name)).href);
const {NodeIO}=await load('@gltf-transform/core');
const {ALL_EXTENSIONS}=await load('@gltf-transform/extensions');
const {dedup,weld,simplify,prune}=await load('@gltf-transform/functions');
const {MeshoptSimplifier}=await load('meshoptimizer');
await MeshoptSimplifier.ready;
const [input,output,ratio='0.08']=process.argv.slice(2);
const io=new NodeIO().registerExtensions(ALL_EXTENSIONS);
const doc=await io.read(resolve(input));
await doc.transform(dedup(),weld(),simplify({simplifier:MeshoptSimplifier,ratio:Number(ratio),error:.03}),prune());
await io.write(resolve(output),doc);
console.log(output);
