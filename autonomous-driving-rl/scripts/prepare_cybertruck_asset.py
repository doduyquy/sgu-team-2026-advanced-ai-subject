"""Split and shade the credited Cybertruck mesh before optional simplification.

Input is standard decoded glTF from optimize_vehicle_assets.mjs. No package
downloads or runtime changes are performed here. Source license is CC BY 4.0.
"""
from pathlib import Path
import argparse,json,struct
import numpy as np


def prepare(source,output):
    data=source.read_bytes();size,_=struct.unpack_from('<II',data,12)
    doc=json.loads(data[20:20+size]);length,_=struct.unpack_from('<II',data,20+size)
    binary=data[28+size:28+size+length]
    def array(index):
        a=doc['accessors'][index];v=doc['bufferViews'][a['bufferView']]
        dtype={5126:'<f4',5125:'<u4',5123:'<u2',5121:'u1'}[a['componentType']]
        columns={'SCALAR':1,'VEC3':3}[a['type']];item=np.dtype(dtype).itemsize
        return np.ndarray((a['count'],columns),dtype=dtype,buffer=binary,
            offset=v.get('byteOffset',0)+a.get('byteOffset',0),
            strides=(v.get('byteStride',item*columns),item)).copy()

    # Three other exported meshes are detached blocks floating above the cab.
    primitive=doc['meshes'][0]['primitives'][0]
    pos=array(primitive['attributes']['POSITION'])
    normals=array(primitive['attributes']['NORMAL'])
    triangles=array(primitive['indices']).reshape(-1,3)
    _,inverse=np.unique(np.round(pos,5),axis=0,return_inverse=True)
    parents=np.arange(int(inverse.max())+1)
    def find(a):
        while parents[a]!=a:parents[a]=parents[parents[a]];a=parents[a]
        return a
    for face in inverse[triangles]:
        a=find(face[0])
        for x in face[1:]:parents[find(x)]=a
    roots=np.array([find(x) for x in inverse])[triangles[:,0]]
    center_x=.3568
    axle_z={'F':1.9636,'R':-1.6794}
    tire_x={'L':-.19575,'R':.90935}
    group_nodes=[];groups={}
    def add(name,material,faces):
        groups.setdefault((name,material),[]).extend(faces)
    for root in np.unique(roots):
        faces=np.flatnonzero(roots==root)
        points=pos[triangles[faces]].reshape(-1,3)
        low,high=points.min(0),points.max(0);center=(low+high)/2;extent=high-low
        axle=min(axle_z,key=lambda key:abs(center[2]-axle_z[key]))
        side='L' if center[0]<center_x else 'R'
        radial=np.hypot(points[:,1]-.4603,points[:,2]-axle_z[axle])
        if high[1]<.97 and low[1]>-.005 and radial.max()<.49 and abs(center[0]-center_x)>.34:
            material='rubber' if radial.max()>.43 else 'wheel_alloy'
            add('Wheel_'+axle+side,material,faces)
            continue
        if extent[1]<.19 and extent[2]<.20 and extent[0]<.10 and len(faces)<6:
            continue
        # Classify closed glass surfaces rather than making all bodywork
        # transparent, which exposed the old model's interior through its roof.
        if center[1]>1.45 and -1.12<center[2]<2.20 and (extent[2]>.45 or extent[0]>.7):
            material='glass'
        elif center[2]<-1.13 and center[1]>1.24:
            material='trim'
        elif center[1]<.62 or (abs(center[0]-center_x)>.61 and center[1]>1.15):
            material='trim'
        elif abs(center[0]-center_x)<.45 and .65<center[1]<1.24 and -1.1<center[2]<1.4:
            material='interior'
        else:material='steel'
        if len(faces)<5000:
            add('Body',material,faces);continue
        # The main shell includes bumpers and fender cladding in one surface.
        for face_index in faces:
            c=pos[triangles[face_index]].mean(0)
            normal=normals[triangles[face_index]].mean(0)
            nearest=min(abs(c[2]-z) for z in axle_z.values())
            distance=np.hypot(c[1]-.4603,nearest)
            trim=(c[1]<.48 or (abs(c[0]-center_x)>.47 and .43<distance<.61)
                  or (c[2]>2.60 and c[1]<.81)
                  or (c[2]<-1.12 and c[1]>1.24 and normal[1]>.25))
            add('Body','trim' if trim else 'steel',[int(face_index)])

    materials=[
        ('steel',[.48,.52,.57,1.],.85,.32),
        ('glass',[.012,.026,.052,1.],.24,.12),
        ('trim',[.018,.022,.028,1.],.12,.55),
        ('interior',[.014,.018,.023,1.],.0,.85),
        ('rubber',[.012,.015,.019,1.],.0,.88),
        ('wheel_alloy',[.018,.025,.034,1.],.35,.35),
    ]
    result={'asset':{'version':'2.0','generator':'Cybertruck race mesh preparation',
        'copyright':'Nieve5677, CC BY 4.0; public copy aditano/tesla-studio; race mesh/material modifications'},
        'scene':0,'scenes':[{'nodes':[]}],'nodes':[],'meshes':[],'materials':[],
        'accessors':[],'bufferViews':[],'buffers':[{'byteLength':0}]}
    for name,color,metal,rough in materials:
        result['materials'].append({'name':name,'pbrMetallicRoughness':{
            'baseColorFactor':color,'metallicFactor':metal,'roughnessFactor':rough}})
    material_ids={m[0]:i for i,m in enumerate(materials)}
    chunk=bytearray()
    def accessor(values,kind):
        while len(chunk)%4:chunk.append(0)
        offset=len(chunk);payload=values.tobytes();chunk.extend(payload)
        view=len(result['bufferViews']);result['bufferViews'].append({'buffer':0,'byteOffset':offset,'byteLength':len(payload)})
        item={'bufferView':view,'componentType':5125 if kind=='SCALAR' else 5126,
              'count':len(values),'type':kind}
        if kind=='VEC3':item.update(min=values.min(0).tolist(),max=values.max(0).tolist())
        result['accessors'].append(item);return len(result['accessors'])-1
    transformed=pos.copy();transformed[:,0]=(pos[:,0]-center_x)*2.177
    transformed[:,2]=-pos[:,2]
    transformed_normals=normals.copy();transformed_normals[:,0]/=2.177
    transformed_normals[:,2]=-normals[:,2]
    transformed_normals/=np.maximum(1e-8,np.linalg.norm(transformed_normals,axis=1))[:,None]
    counts={}
    for node in ('Body','Wheel_FL','Wheel_FR','Wheel_RL','Wheel_RR'):
        primitives=[]
        for (name,material),faces in groups.items():
            if name!=node:continue
            tri=triangles[np.asarray(faces)]
            if name=='Body':
                # The source has curved/smoothed door panels. The reference
                # has broad planar steel: flatten its outer side band and use
                # geometric face normals so reflections do not look crumpled.
                positions=pos.copy()
                side_band=(np.abs(positions[:,0]-center_x)>.46)&(positions[:,1]>.50)&(positions[:,1]<1.32)&(positions[:,2]>-1.14)&(positions[:,2]<1.45)
                positions[side_band,0]=center_x+np.sign(positions[side_band,0]-center_x)*(.64-.065*(positions[side_band,1]-.5))
                positions[:,0]=(positions[:,0]-center_x)*2.177;positions[:,2]*=-1
                points=positions[tri[:,[0,2,1]]].reshape(-1,3)
                faces3=points.reshape(-1,3,3)
                flat=np.cross(faces3[:,1]-faces3[:,0],faces3[:,2]-faces3[:,0])
                flat/=np.maximum(1e-8,np.linalg.norm(flat,axis=1))[:,None]
                normals3=np.repeat(np.round(flat,5),3,axis=0)
                primitives.append({'attributes':{'POSITION':accessor(points.astype('<f4'),'VEC3'),
                    'NORMAL':accessor(normals3.astype('<f4'),'VEC3')},
                    'indices':accessor(np.arange(len(points),dtype='<u4'),'SCALAR'),
                    'material':material_ids[material]})
                counts[node]=counts.get(node,0)+len(tri)
                continue
            indices,remap=np.unique(tri.reshape(-1),return_inverse=True)
            # Z reflection reverses winding, so restore outward front faces.
            remap=remap.reshape(-1,3)[:,[0,2,1]].reshape(-1).astype('<u4')
            primitives.append({'attributes':{'POSITION':accessor(transformed[indices].astype('<f4'),'VEC3'),
                'NORMAL':accessor(transformed_normals[indices].astype('<f4'),'VEC3')},
                'indices':accessor(remap,'SCALAR'),'material':material_ids[material]})
            counts[node]=counts.get(node,0)+len(tri)
        result['meshes'].append({'name':node,'primitives':primitives})
        result['nodes'].append({'name':node,'mesh':len(result['meshes'])-1})
        result['scenes'][0]['nodes'].append(len(result['nodes'])-1)
    result['buffers'][0]['byteLength']=len(chunk)
    encoded=json.dumps(result,separators=(',',':')).encode()
    encoded+=b' '*((-len(encoded))%4);chunk+=b'\0'*((-len(chunk))%4)
    total=12+8+len(encoded)+8+len(chunk)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(struct.pack('<III',0x46546c67,2,total)+struct.pack('<II',len(encoded),0x4e4f534a)
                       +encoded+struct.pack('<II',len(chunk),0x004e4942)+chunk)
    print(json.dumps({'triangles':counts,'bytes':total},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    arguments=parser.parse_args();prepare(arguments.source,arguments.output)
