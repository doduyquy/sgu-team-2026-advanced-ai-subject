"""Batched riverside boulevard scenery with local LOD for the VN corridor.

The road follows the bundled OSM carriageway; buildings, canal treatment,
trees and signs are authored scenery, not a surveyed city reconstruction.
"""
from array import array
import math
from pathlib import Path
import random

import numpy as np
from panda3d.core import (Filename, Geom, GeomNode, GeomTriangles, GeomVertexData,
                         GeomVertexFormat, GeomVertexArrayFormat, InternalName,
                         LODNode, NodePath, Shader, TextNode, LineSegs)
from src.visualization.vietnam_track import road_frame, route_data
from src.visualization.vietnam_assets import ROOT as SCENERY_ROOT, tree, facade, tree_shadow

_VERTEX = '''#version 330
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ModelMatrix;
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec4 p3d_Color;
in vec2 p3d_MultiTexCoord0;
out vec4 shade;
out vec2 uv;
out float distance_to_eye;
out vec3 world;
out vec3 world_normal;
void main() {
    gl_Position=p3d_ModelViewProjectionMatrix*p3d_Vertex;
    shade=p3d_Color;
    world=(p3d_ModelMatrix*p3d_Vertex).xyz;
    world_normal=normalize(mat3(p3d_ModelMatrix)*p3d_Normal);
    uv=p3d_MultiTexCoord0;
    distance_to_eye=length((p3d_ModelViewMatrix*p3d_Vertex).xyz);
}'''
_FRAGMENT = '''#version 330
uniform sampler2D surface_texture;
uniform sampler2D road_normal;
uniform int surface_mode;
uniform vec3 camera_world_position;
uniform float vn_time;
in vec4 shade;
in vec2 uv;
in float distance_to_eye;
in vec3 world;
in vec3 world_normal;
out vec4 color;
void main() {
    vec3 rgb=shade.rgb;
    vec3 n=normalize(world_normal);
    if(surface_mode==1 || surface_mode==3 || surface_mode==4) rgb*=texture(surface_texture,uv).rgb;
    if(surface_mode==5) rgb*=mix(vec3(1.),texture(surface_texture,uv).rgb,.30);
    if(surface_mode==4) {
        vec3 detail=texture(road_normal,uv).rgb*2.-1.;
        n=normalize(vec3(detail.xy*.32,1.));
    }
    vec3 sun=normalize(vec3(-.35,-.45,.82));
    rgb*=.55+.45*max(0.,dot(n,sun));
    if(surface_mode==2) {
        vec2 p=world.xy;
        n=normalize(vec3(.035*sin(p.x*2.1+p.y*.8+vn_time*.7),
                         .045*cos(p.y*2.7-p.x*.6+vn_time*.55),1.));
        vec3 eye=normalize(camera_world_position-world);
        float fresnel=.035+.65*pow(1.-max(0.,dot(n,eye)),5.);
        vec3 reflection=mix(vec3(.28,.36,.36),vec3(.47,.59,.65),max(0.,reflect(-eye,n).z));
        float glint=pow(max(0.,dot(reflect(-sun,n),eye)),180.);
        rgb=mix(vec3(.045,.105,.098),reflection,fresnel)+vec3(.8,.74,.58)*glint*.38;
    }
    float fog=smoothstep(300.,1100.,distance_to_eye);
    color=vec4(mix(rgb,vec3(.56,.62,.63),fog*.6),shade.a);
}'''


class MeshBatch:
    def __init__(self):
        self.values = array('f')
        self.indices = []

    def face(self, points, color, normal=(0, 0, 1), uv=None):
        start = len(self.values)//12
        for i, point in enumerate(points):
            if uv is not None:
                coords=uv[i]
            elif abs(normal[2])>.7:
                coords=(point[0]/3,point[1]/3)
            elif abs(normal[1])>.7:
                coords=(point[0]/3,point[2]/3)
            else:
                coords=(point[1]/3,point[2]/3)
            self.values.extend((*point, *normal, *color, *coords))
        for i in range(1, len(points)-1):
            self.indices.extend((start, start+i, start+i+1))

    def box(self, center, size, color):
        x, y, z = center
        a, b, c = (v/2 for v in size)
        vertices = [(x+sx*a, y+sy*b, z+sz*c) for sx, sy, sz in
                    ((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                     (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1))]
        for face, normal in (((0,3,2,1),(0,0,-1)),((4,5,6,7),(0,0,1)),
                             ((0,1,5,4),(0,-1,0)),((1,2,6,5),(1,0,0)),
                             ((2,3,7,6),(0,1,0)),((3,0,4,7),(-1,0,0))):
            self.face([vertices[i] for i in face], color, normal)

    def canopy(self, center, radius, height, color, sides=8):
        x,y,z = center
        rings = []
        for dz, scale in ((0,.58),(height*.4,1.),(height*.78,.7),(height,0.)):
            rings.append([(x+radius*scale*math.cos(i*math.tau/sides),
                           y+radius*scale*math.sin(i*math.tau/sides),z+dz) for i in range(sides)])
        for a,b in zip(rings[:-1], rings[1:]):
            for i in range(sides):
                j=(i+1)%sides
                self.face((a[i],a[j],b[j],b[i]),color,
                          (math.cos(i*math.tau/sides)*.5,math.sin(i*math.tau/sides)*.5,.7))

    def attach(self, parent, name):
        layout=GeomVertexArrayFormat()
        for column,size,contents in ((InternalName.getVertex(),3,Geom.CPoint),
                                     (InternalName.getNormal(),3,Geom.CNormal),
                                     (InternalName.getColor(),4,Geom.CColor),
                                     (InternalName.getTexcoord(),2,Geom.CTexcoord)):
            layout.addColumn(column,size,Geom.NTFloat32,contents)
        format_=GeomVertexFormat();format_.addArray(layout)
        data = GeomVertexData(name, GeomVertexFormat.registerFormat(format_), Geom.UHStatic)
        data.setNumRows(len(self.values)//12)
        memoryview(data.modifyArray(0)).cast('B').cast('f')[:] = self.values
        primitive = GeomTriangles(Geom.UHStatic)
        for i in range(0,len(self.indices),3):
            primitive.addVertices(*self.indices[i:i+3])
        geometry = Geom(data)
        geometry.addPrimitive(primitive)
        node = GeomNode(name)
        node.addGeom(geometry)
        path = parent.attachNewNode(node)
        path.setTwoSided(True)
        return path


def _strip(parent, points, right, a, b, z, color, name, texture=None, mode=0):
    batch = MeshBatch()
    left, edge = points+right*a, points+right*b
    for i in range(len(points)-1):
        quad = [(float(p[0]),float(p[1]),z) for p in (left[i],edge[i],edge[i+1],left[i+1])]
        batch.face(quad, color)
    path = batch.attach(parent,name)
    path.setShaderInput('surface_mode',mode)
    if texture is not None:
        path.setShaderInput('surface_texture',texture)
    return path


def _text(parent, font, text, position, scale, color=(1,1,1,1), heading=-90):
    node = TextNode('vn_sign_text')
    node.setText(text)
    node.setFont(font)
    node.setAlign(TextNode.ACenter)
    node.setTextColor(*color)
    root = parent.attachNewNode(node)
    root.setPos(*position)
    root.setH(heading)
    root.setScale(scale)
    root.setShaderOff()
    root.setLightOff()
    return root


class VietnamScenery:
    def __init__(self, terrain, course):
        from metadrive.constants import CamMask
        self.engine = terrain.engine
        self.root = self.engine.render.attachNewNode('vietnam_boulevard')
        self.root.hide(CamMask.AllOn)
        self.root.show(CamMask.MainCam|CamMask.RgbCam)
        self.root.setShader(Shader.make(Shader.SL_GLSL,vertex=_VERTEX,fragment=_FRAGMENT))
        self.root.setShaderInput('surface_mode',0)
        self.root.setShaderInput('surface_texture',terrain.road_texture)
        self.root.setShaderInput('road_normal',terrain.road_texture_normal)
        self.root.setShaderInput('vn_time',0.)
        self.shader=terrain.engine._race_surface_shader
        self.font = self.engine.loader.loadFont(Filename.fromOsSpecific('C:/Windows/Fonts/segoeuib.ttf').getFullpath())
        self.paving=self.engine.loader.loadTexture(Filename.fromOsSpecific(str(SCENERY_ROOT/'brick_pavement/brick_pavement_diff_1k.jpg')).getFullpath())
        self.plaster=self.engine.loader.loadTexture(Filename.fromOsSpecific(str(SCENERY_ROOT/'grey_plaster/grey_plaster_diff_1k.jpg')).getFullpath())
        for texture in (self.paving,self.plaster):
            texture.setFormat(texture.FSrgb);texture.setAnisotropicDegree(8)
        points = course.route_points
        _,right = road_frame(points)
        # Full-corridor meshes never rely on the simulator's 2 km raster.
        _strip(self.root,points,right,-240,240,-.12,(.12,.21,.09,1),'vn_green_ground',terrain.grass_tex,3)
        _strip(self.root,points,right,-6.5,6.5,.005,(.40,.425,.44,1),'vn_asphalt',terrain.road_texture,4)
        _strip(self.root,points,right,-26.5,-13.5,.005,(.40,.425,.44,1),'vn_opposite_boulevard',terrain.road_texture,4)
        for a,b in ((-13.5,-6.5),(6.5,11.5),(-31,-26.5)):
            _strip(self.root,points,right,a,b,.16,(1.05,1.08,1.02,1),'vn_promenade',self.paving,1)
        for edge in (-26.5,-13.5,-6.5,6.5):
            _strip(self.root,points,right,edge-.15,edge+.15,.18,(.68,.68,.62,1),'vn_kerb_top',self.plaster,1)
            curb=MeshBatch()
            for i in range(len(points)-1):
                a,b=points[i]+right[i]*edge,points[i+1]+right[i+1]*edge
                curb.face([(float(p[0]),float(p[1]),z) for p,z in ((a,.005),(b,.005),(b,.18),(a,.18))],(.58,.58,.52,1),(*right[i],0))
            curb.attach(self.root,'vn_raised_kerb')
        _strip(self.root,points,right,-12.5,-7.5,.17,(.12,.24,.10,1),'vn_planted_median',terrain.grass_tex,3)
        _strip(self.root,points,right,-33,-31,.08,(.66,.65,.56,1),'vn_canal_embankment')
        _strip(self.root,points,right,-78,-33,-.1,(.035,.22,.25,1),'vn_canal',mode=2)
        _strip(self.root,points,right,-81,-78,.08,(.66,.65,.56,1),'vn_far_embankment')
        _strip(self.root,points,right,-89,-81,.02,(.32,.38,.35,1),'vn_riverside_walk')
        # Thin shoulders, solid edge lines, and 2 m broken lane markings.
        for offset in (-5.25,5.25,-25.25,-14.75):
            _strip(self.root,points,right,offset-.065,offset+.065,.025,(.94,.94,.86,1),'vn_edge_paint')
        for offset in (-1.75,1.75,-18.25,-21.75):
            batch = MeshBatch()
            for i in range(0,len(points)-1,2):
                a,b = points[i],points[i+1]
                batch.face([(float(p[0]),float(p[1]),.026) for p in
                            (a+right[i]*(offset-.06),a+right[i]*(offset+.06),
                             b+right[i+1]*(offset+.06),b+right[i+1]*(offset-.06))],(.92,.92,.86,1))
            batch.attach(self.root,'vn_broken_lane_paint')
        line=LineSegs('vn_minimap_route')
        line.setThickness(3.5);line.setColor(.98,.74,.20,1)
        line.moveTo(float(points[0,0]),float(points[0,1]),.2)
        for point in points[::5]:line.drawTo(float(point[0]),float(point[1]),.2)
        self.minimap=terrain.engine.render.attachNewNode(line.create())
        self.minimap.setShaderOff();self.minimap.setLightOff()
        self.minimap.hide(CamMask.AllOn);self.minimap.show(CamMask.MiniMap)
        self._build_chunks(points,right)

    def _build_chunks(self, points, right):
        rng = random.Random(2026)
        cumulative = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
        for distance in np.arange(0,cumulative[-1],160):
            index = min(len(points)-2,int(np.searchsorted(cumulative,distance)))
            origin=points[index]
            tangent=points[index+1]-origin
            heading=math.degrees(math.atan2(tangent[1],tangent[0]))
            lod=LODNode('vn_streetscape_lod')
            node=self.root.attachNewNode(lod)
            node.setPos(float(origin[0]),float(origin[1]),0)
            node.setH(heading)
            lod.addSwitch(700,0)
            chunk=node.attachNewNode('vn_streetscape')
            batch=MeshBatch()
            soil=MeshBatch()
            def local(forward,lateral):
                idx=min(len(points)-1,int(np.searchsorted(cumulative,distance+forward)))
                world=points[idx]+right[idx]*lateral-origin
                theta=math.radians(heading)
                return (float(world[0]*math.cos(theta)+world[1]*math.sin(theta)),
                        float(-world[0]*math.sin(theta)+world[1]*math.cos(theta)))
            for x in range(0,160,20):
                # Cream and pastel townhouses, inset glass, balconies and awnings.
                for side in (-1,1):
                    bx,y=local(x,28 if side<0 else -104)
                    height=rng.choice((8.,11.,14.,17.))
                    color=rng.choice(((.64,.54,.38,1),(.55,.64,.61,1),(.66,.64,.56,1),(.54,.59,.66,1)))
                    batch.box((bx,y,height/2),(13,12,height),tuple(c*.78 for c in color[:3])+(1,))
                    batch.box((bx,y,height+.2),(14,13,.4),(.25,.30,.30,1))
                    for floor in np.arange(3,height,3):
                        for dx in (-4,0,4):
                            batch.box((bx+dx,y+(6.05 if side<0 else -6.05),float(floor)),
                                      (2.4,.08,1.5),(.055,.15,.21,1))
                        batch.box((bx,y+(6.5 if side<0 else -6.5),float(floor)-1),(12,1.4,.18),(.65,.65,.56,1))
                    batch.box((bx,y+(7.2 if side<0 else -7.2),2.5),(12,2.6,.15),(.08,.28,.20,1))
                    facade(chunk,self.shader,(bx,y+(6.2 if side<0 else -6.2),0),height,
                           variant=(x//20+int(distance)//160)%3,heading=180 if side<0 else 0)
                    if side<0 and x==40:
                        _text(chunk,self.font,rng.choice(('CÀ PHÊ SÀI GÒN','PHỞ VIỆT','BÁNH MÌ SÀI GÒN')),
                              (bx,y+6.2,2.7),.60,(.96,.82,.45,1),heading=180)
                for y in (-12.,9.,85.):
                    tx,ty=local(x,-y)
                    scale=rng.uniform(1.7,2.3)
                    rotation=rng.uniform(0,360)
                    tree(chunk,self.shader,(tx,ty,0),scale,rotation)
                    tree_shadow(chunk,(tx,ty,0),scale,rotation)
                    if y == -12.:
                        soil.face([(tx+dx,ty+dy,.172) for dx,dy in ((-1,-1),(1,-1),(1,1),(-1,1))],(.10,.075,.045,1))
            # Modern lamps stay clear of all drivable lanes.
            for x in (8,48,88,128):
                lx,ly=local(x,8)
                batch.box((lx,ly,4.4),(.15,.15,8.8),(.23,.26,.27,1))
                batch.box((lx,ly+2.5,8.7),(.14,5.,.14),(.23,.26,.27,1))
                batch.box((lx,ly+4.7,8.65),(.65,1.1,.13),(.95,.88,.60,1))
            # Palm-like canopies on the median and stylized distant towers.
            if int(distance)%480 == 0:
                for x,y,height in ((30,-140,38),(90,-160,54),(130,-130,29)):
                    batch.box((x,y,height/2),(16,20,height),(.31,.42,.48,1))
                    for z in range(5,height,4):
                        batch.box((x,y+10.05,z),(13,.1,1.1),(.13,.25,.30,1))
            buildings=batch.attach(chunk,'vn_batched_street_furniture')
            soil.attach(chunk,'vn_tree_soil_beds')
            buildings.setShaderInput('surface_mode',5)
            buildings.setShaderInput('surface_texture',self.plaster)
            if distance == 0:
                self._gantry(chunk,35,'VIỆT NAM  /  SÀI GÒN','VÕ VĂN KIỆT  ·  10 KM',start=True)
            elif 480 <= distance < 640 or int(distance)%960 == 0:
                km=int(round(distance/1000))
                self._gantry(chunk,40,f'VÕ VĂN KIỆT   ·   KM {km:02}',
                             f'ĐÍCH  {max(0,(cumulative[-1]-distance)/1000):.1f} KM   ↑')
            if distance+160 >= cumulative[-1]:
                self._gantry(chunk,70,'ĐÍCH ĐẾN  /  FINISH','VIỆT NAM ENDURANCE')

    def _gantry(self,parent,x,title,detail,start=False):
        batch=MeshBatch()
        for y in (-7.2,7.2):
            batch.box((x,y,3.8),(.35,.35,7.6),(.35,.40,.42,1))
        batch.box((x,0,7.6),(.35,14.8,.35),(.35,.40,.42,1))
        batch.box((x,0,6.4),(.26,12.4,2.2),(.025,.22,.18,1))
        batch.box((x-.15,0,7.45),(.02,12.4,.08),(.84,.73,.29,1))
        if start:
            batch.box((x-.18,5.35,6.4),(.02,1.5,1.1),(.70,.018,.035,1))
            star=[]
            for i in range(10):
                angle=math.pi/2+i*math.pi/5
                radius=.46 if i%2==0 else .19
                star.append((x-.2,5.35+radius*math.cos(angle),6.4+radius*math.sin(angle)))
            for i in range(10):
                batch.face(((x-.2,5.35,6.4),star[i],star[(i+1)%10]),(.98,.76,.13,1),(-1,0,0))
        batch.attach(parent,'vn_gantry')
        _text(parent,self.font,title,(x-.16,0,6.65),.68,(.96,.92,.74,1))
        _text(parent,self.font,detail,(x-.16,0,5.85),.43)

    def destroy(self):
        self.root.removeNode()
        self.minimap.removeNode()


def update_vietnam_scenery(terrain):
    course=terrain.engine.current_map
    active=bool(getattr(course,'is_vietnam',False))
    previous=getattr(terrain,'_vietnam_scenery',None)
    if previous:
        previous.destroy()
        terrain._vietnam_scenery=None
    from metadrive.constants import CamMask
    mask=CamMask.MainCam|CamMask.RgbCam
    terrain.origin.hide(mask) if active else terrain.origin.show(mask)
    if active:
        terrain._vietnam_scenery=VietnamScenery(terrain,course)
    return active
