"""Bake CC0 original tree geometry into transparent far-LOD cards (offline)."""
from pathlib import Path
import math
from PySide6.QtCore import QProcess  # DLL order
from panda3d.core import (loadPrcFileData, OrthographicLens, NodePath, Shader,
                         TransparencyAttrib, Filename)
from direct.showbase.ShowBase import ShowBase
from gltf import load_model

ROOT=Path(__file__).resolve().parents[1]/'assets/scenery/island_tree_02'
loadPrcFileData('tree_bake','window-type offscreen\nwin-size 768 768\nframebuffer-alpha true\nframebuffer-multisample true\nmultisamples 4\nbackground-color 0 0 0 0\nsync-video false')
app=ShowBase(windowType='offscreen')
tree=NodePath(load_model(str(ROOT/'island_tree_02_1k.gltf')))
tree.reparentTo(app.render)
vertex='''#version 330
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
in vec4 p3d_Vertex;in vec3 p3d_Normal;in vec2 p3d_MultiTexCoord0;
out vec3 normal;out vec2 uv;
void main(){gl_Position=p3d_ModelViewProjectionMatrix*p3d_Vertex;normal=mat3(p3d_ModelMatrix)*p3d_Normal;uv=p3d_MultiTexCoord0;}
'''
fragment='''#version 330
uniform sampler2D p3d_Texture0;
in vec3 normal;in vec2 uv;out vec4 color;
void main(){vec4 tex=texture(p3d_Texture0,uv);if(tex.a<.4)discard;
float light=.45+.55*max(0.,dot(normalize(normal),normalize(vec3(-.3,-.5,1.))));
color=vec4(pow(tex.rgb*light,vec3(1./2.2)),1.);}
'''
tree.setShader(Shader.make(Shader.SL_GLSL,vertex=vertex,fragment=fragment),30)
tree.setTransparency(TransparencyAttrib.MBinary,50)
lens=OrthographicLens();lens.setFilmSize(5.4,5.4);lens.setNearFar(.1,100)
app.cam.node().setLens(lens)
for index,angle in enumerate((0,90,180)):
    a=math.radians(angle)
    app.camera.setPos(10*math.sin(a),-10*math.cos(a),2.7)
    app.camera.lookAt(0,-.7,1.7)
    for _ in range(4):app.graphicsEngine.renderFrame()
    output=ROOT/f'tree-card-{index}.png'
    app.win.saveScreenshot(Filename.fromOsSpecific(str(output)))
    print(output,flush=True)
app.destroy()
