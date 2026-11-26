"""Photographic CC0 scenery assets with shared meshes, PBR and distance LOD."""
from array import array
from functools import lru_cache
import math
from pathlib import Path
import random

from panda3d.core import (AlphaTestAttrib, LODNode, Material, NodePath, Shader,
                         Texture, TransparencyAttrib)

ROOT=Path(__file__).resolve().parents[2]/'assets/scenery'


@lru_cache(maxsize=1)
def daylight_environment():
    from simplepbr.envmap import EnvMap
    size=24
    values=array('f')
    for face in range(6):
        for y in range(size):
            v=2*(y+.5)/size-1
            for x in range(size):
                u=2*(x+.5)/size-1
                direction=((1,-v,-u),(-1,-v,u),(u,1,v),(u,-1,-v),(u,-v,1),(u,v,-1))[face]
                length=math.sqrt(sum(d*d for d in direction))
                dx,dy,dz=(d/length for d in direction)
                sky=max(0.,dz)
                sun=max(0.,dx*.35-dy*.45+dz*.82)**96*7.
                if dz>=0:
                    r,g,b=.23+.10*sky+sun,.31+.15*sky+sun*.93,.39+.23*sky+sun*.77
                else:
                    r,g,b=.11,.13,.085
                values.extend((b,g,r))
    cube=Texture('vn_daylight_environment')
    cube.setupCubeMap(size,Texture.TFloat,Texture.FRgb32)
    cube.setRamImage(values.tobytes())
    return EnvMap(cube,prefiltered_size=32,prefiltered_samples=8,blocking_prepare=True)


def shade(root, shader):
    from simplepbr import Pipeline
    environment=daylight_environment()
    root.setShader(shader,20)
    root.setShaderInput('sh_coeffs',environment.sh_coefficients)
    root.setShaderInput('brdf_lut',Pipeline._BRDF_LUT)
    root.setShaderInput('filtered_env_map',environment.filtered_env_map)
    root.setShaderInput('max_reflection_lod',environment.filtered_env_map.num_loadable_ram_mipmap_images)
    root.setShaderInput('global_shadow_bias',.005)


@lru_cache(maxsize=8)
def _model(relative):
    from gltf import load_model
    return NodePath(load_model(str(ROOT/relative)))


def tree(parent, shader, position, scale=1., heading=0):
    from src.visualization.race_graphics import get_graphics_profile
    near_distance,far_distance=(42,190) if get_graphics_profile().name=='lite' else (55,250)
    root=parent.attachNewNode('vn_photographic_tree')
    root.setPos(*position);root.setH(heading)
    lod=root.attachNewNode(LODNode('vn_tree_lod'))
    # Keep the LOD node unscaled: a 2x tree must not double its detail radius.
    for file,near,far in (('tree-near-v2.glb',0,near_distance),('tree-far.glb',near_distance,far_distance)):
        level=lod.attachNewNode('vn_tree_level')
        level.setScale(scale)
        _model('island_tree_02/'+file).instanceTo(level)
        _leaf_cloud(2800 if near==0 else 450).instanceTo(level)
        lod.node().addSwitch(far,near)
        shade(level,shader)
        level.setTransparency(TransparencyAttrib.MBinary,30)
        level.setAttrib(AlphaTestAttrib.make(AlphaTestAttrib.MGreaterEqual,.35),30)
    return root


_LEAF_VERTEX='''#version 330
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
in vec4 p3d_Vertex;in vec3 p3d_Normal;in vec4 p3d_Color;in vec2 p3d_MultiTexCoord0;
out vec3 normal;out vec4 tint;out vec2 uv;
void main(){gl_Position=p3d_ModelViewProjectionMatrix*p3d_Vertex;normal=mat3(p3d_ModelMatrix)*p3d_Normal;uv=p3d_MultiTexCoord0;tint=p3d_Color;}
'''
_LEAF_FRAGMENT='''#version 330
uniform sampler2D leaf_texture;
in vec3 normal;in vec4 tint;in vec2 uv;out vec4 color;
void main(){vec4 leaf=texture(leaf_texture,uv);if(leaf.a<.4)discard;
float sun=dot(normalize(normal),normalize(vec3(-.35,-.45,.82)));
float light=.48+.48*max(0.,sun)+.16*max(0.,-sun);
color=vec4(leaf.rgb*tint.rgb*light,1.);}
'''


@lru_cache(maxsize=3)
def _leaf_cloud(count):
    import numpy as np
    from src.visualization.vietnam_scenery import MeshBatch
    batch=MeshBatch();rng=random.Random(835+count)
    texture=next(t for t in _model('island_tree_02/tree-near-v2.glb').findAllTextures() if 'leaves_diff' in t.getName())
    # UV rectangles isolate individual photographed leaves from the CC0 atlas.
    regions=((.02,.49,.20,.99),(.20,.49,.38,.99),(.39,.49,.57,.99),
             (.58,.49,.77,.99),(.78,.49,.99,.99),(.02,.02,.21,.48),
             (.22,.02,.43,.48),(.44,.02,.66,.48))
    multiplier=1. if count>1000 else math.sqrt(2800/count)
    for _ in range(count):
        azimuth=rng.random()*math.tau
        vertical=rng.uniform(-1,1)
        radius=rng.random()**.35
        ring=math.sqrt(max(0,1-vertical*vertical))
        center=np.array((2.0*ring*math.cos(azimuth)*radius,
                         -.8+1.9*ring*math.sin(azimuth)*radius,2.35+1.05*vertical*radius))
        normal=np.array((rng.uniform(-1,1),rng.uniform(-1,1),rng.uniform(-.5,1)))
        normal/=np.linalg.norm(normal)
        across=np.cross(normal,(0,0,1))
        across/=max(np.linalg.norm(across),1e-6)
        along=np.cross(normal,across)
        width=rng.uniform(.10,.17)*multiplier
        height=rng.uniform(.22,.34)*multiplier
        a=across*width/2;b=along*height/2
        u0,v0,u1,v1=rng.choice(regions)
        tint=(rng.uniform(.72,1.),rng.uniform(.88,1.12),rng.uniform(.62,.88),1)
        batch.face([tuple(p) for p in (center-a-b,center+a-b,center+a+b,center-a+b)],
                   tint,tuple(normal),((u0,v0),(u1,v0),(u1,v1),(u0,v1)))
    root=NodePath('vn_photographic_leaf_cloud')
    batch.attach(root,'vn_photographic_leaf_cards')
    root.setShader(Shader.make(Shader.SL_GLSL,vertex=_LEAF_VERTEX,fragment=_LEAF_FRAGMENT),35)
    root.setShaderInput('leaf_texture',texture)
    root.setTexture(texture)
    root.setTransparency(TransparencyAttrib.MBinary,35)
    return root


@lru_cache(maxsize=3)
def _facade_template(variant):
    root=NodePath('vn_facade_template')
    for floor in range(3+variant%2):
        for column in range(3):
            kind='door' if floor==0 and column==1 else ('large' if (variant+column)%2 else 'small')
            transform=root.attachNewNode('vn_wall_unit')
            transform.setPos((column-1)*3+1.5,0,floor*3)
            _model('modular_urban_apartments_facade/facade-'+kind+'.glb').instanceTo(transform)
    roof=root.attachNewNode('vn_cornice')
    roof.setPos(-1.5,0,(3+variant%2)*3)
    for column in range(3):
        part=roof.attachNewNode('vn_roof_trim');part.setX(column*3)
        _model('modular_urban_apartments_facade/facade-cornice.glb').instanceTo(part)
    root.flattenStrong()
    return root


def facade(parent,shader,position,height,variant=0,heading=0):
    root=parent.attachNewNode('vn_detailed_facade')
    root.setPos(*position);root.setH(heading)
    lod=root.attachNewNode(LODNode('vn_facade_lod'))
    detail=lod.attachNewNode('vn_facade_near')
    detail.setScale(13/9,1,height/((3+variant%2)*3))
    _facade_template(variant%3).instanceTo(detail)
    shade(detail,shader)
    lod.node().addSwitch(120,0)
    return root


_SHADOW_VERTEX='''#version 330
uniform mat4 p3d_ModelMatrix;
uniform mat4 p3d_ViewProjectionMatrix;
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
out vec2 uv;
void main(){
 vec4 world=p3d_ModelMatrix*p3d_Vertex;
 world.xy+=world.z*vec2(.65,-.4);
 world.z=.055;
 gl_Position=p3d_ViewProjectionMatrix*world;
 uv=p3d_MultiTexCoord0;
}'''
_SHADOW_FRAGMENT='''#version 330
uniform sampler2D p3d_Texture0;
in vec2 uv;
out vec4 color;
void main(){
 float alpha=texture(p3d_Texture0,uv).a;
 if(alpha<.4)discard;
 color=vec4(.015,.025,.035,.13);
}'''


def tree_shadow(parent,position,scale=1.,heading=0):
    root=parent.attachNewNode('vn_tree_shadow')
    root.setPos(*position);root.setH(heading)
    lod=root.attachNewNode(LODNode('vn_shadow_lod'))
    detail=lod.attachNewNode('vn_shadow_geometry')
    detail.setScale(scale)
    _model('island_tree_02/tree-far.glb').instanceTo(detail)
    _leaf_cloud(450).instanceTo(detail)
    lod.node().addSwitch(65,0)
    root.setShader(Shader.make(Shader.SL_GLSL,vertex=_SHADOW_VERTEX,fragment=_SHADOW_FRAGMENT),50)
    root.setTransparency(TransparencyAttrib.MAlpha,50)
    root.setDepthWrite(False)
    root.setDepthOffset(1)
    return root
