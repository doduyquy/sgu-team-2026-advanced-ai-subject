import {createRequire} from 'node:module';
import {pathToFileURL,fileURLToPath} from 'node:url';
const require=createRequire(new URL('../artifacts/vehicle-asset-tools/package.json',import.meta.url));
const load=async name=>import(pathToFileURL(require.resolve(name)).href);
const {NodeIO}=await load('@gltf-transform/core');
const {ALL_EXTENSIONS}=await load('@gltf-transform/extensions');
const {prune,dedup}=await load('@gltf-transform/functions');
const io=new NodeIO().registerExtensions(ALL_EXTENSIONS);
const folder=fileURLToPath(new URL('../assets/scenery/modular_urban_apartments_facade',import.meta.url));
const units={
  large:['wall_window_centered_large_01','window_centered_large_01'],
  small:['wall_window_centered_small_01','window_centered_small_01'],
  door:['wall_door_centered_large_01','door_centered_large_01'],
  cornice:['cornice_standard_standard_01'],
};
for(const [name,keep] of Object.entries(units)){
 const doc=await io.read(folder+'/modular_urban_apartments_facade_1k.gltf');
 for(const scene of doc.getRoot().listScenes())for(const node of scene.listChildren()){
  if(!keep.includes(node.getName()))scene.removeChild(node);else node.setTranslation([0,0,0]);
 }
 await doc.transform(prune(),dedup());
 await io.write(folder+'/facade-'+name+'.glb',doc);
 console.log(name);
}
