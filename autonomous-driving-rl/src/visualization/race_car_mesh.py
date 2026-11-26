"""Small original car meshes with smooth normals and shared PBR materials.

All coordinates are render-only, in the existing vehicle's local frame.
No collision bodies, suspension settings or manufacturer CAD are included.
"""

import math
from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexData,
                         GeomVertexFormat, GeomVertexWriter, NodePath, Vec3)
from src.visualization.race_style import paint_material


def mesh(parent, name, vertices, faces, color):
    normals = [Vec3(0, 0, 0) for _ in vertices]
    for face in faces:
        a, b, c = (Vec3(*vertices[i]) for i in face[:3])
        normal = (b-a).cross(c-a)
        for i in face:
            normals[i] += normal
    data = GeomVertexData(name, GeomVertexFormat.getV3n3c4(), Geom.UHStatic)
    pos, norm, shade = [GeomVertexWriter(data, column) for column in ('vertex', 'normal', 'color')]
    primitive = GeomTriangles(Geom.UHStatic)
    for point, normal in zip(vertices, normals):
        normal.normalize()
        pos.addData3(*point)
        norm.addData3(normal)
        shade.addData4(*color)
    for face in faces:
        for i in range(1, len(face)-1):
            primitive.addVertices(face[0], face[i], face[i+1])
    geom = Geom(data)
    geom.addPrimitive(primitive)
    node = GeomNode(name)
    node.addGeom(geom)
    return parent.attachNewNode(node)


def _interpolate(sections, steps=3):
    result = []
    for i in range(len(sections)-1):
        a, b = sections[max(0, i-1)], sections[i]
        c, d = sections[i+1], sections[min(len(sections)-1, i+2)]
        for j in range(steps):
            t = j/steps
            result.append(tuple(.5*((2*b[k])+(-a[k]+c[k])*t+
                                (2*a[k]-5*b[k]+4*c[k]-d[k])*t*t+
                                (-a[k]+3*b[k]-3*c[k]+d[k])*t*t*t)
                                for k in range(4)))
    return result + [sections[-1]]


def loft(parent, name, sections, color, curved=True):
    """Rounded rectangular cross sections, not a stack of box primitives."""
    if curved:
        sections = _interpolate(sections)
    ring = ((.72,0),(.94,.035),(1,.16),(1,.65),(.94,.85),(.72,1),
            (-.72,1),(-.94,.85),(-1,.65),(-1,.16),(-.94,.035),(-.72,0))
    vertices = [(x*w, y, lo+(hi-lo)*z) for y,w,lo,hi in sections for x,z in ring]
    n = len(ring)
    faces = [tuple(range(n)), tuple(reversed(range(len(vertices)-n, len(vertices))))]
    for i in range(len(sections)-1):
        for j in range(n):
            k = (j+1)%n
            faces.append((i*n+j,(i+1)*n+j,(i+1)*n+k,i*n+k))
    return mesh(parent, name, vertices, faces, color)


def strip(parent, name, points, width, color):
    """A narrow raised trim line, with smooth highlights along the curve."""
    sections = []
    for x,y,z in points:
        sections.append(((x-width/2,y,z),(x+width/2,y,z),
                         (x+width/2,y,z+width/2),(x-width/2,y,z+width/2)))
    vertices = [p for section in sections for p in section]
    faces = []
    for i in range(len(sections)-1):
        for j in range(4):
            faces.append((i*4+j,(i+1)*4+j,(i+1)*4+(j+1)%4,i*4+(j+1)%4))
    return mesh(parent, name, vertices, faces, color)


def panel(parent, name, points, color):
    # Separate front/back faces; no renderer-wide two-sided drawing required.
    n = len(points)
    return mesh(parent, name, list(points)+list(points),
                [tuple(range(n)), tuple(reversed(range(n,2*n)))], color)


def _groups(root):
    groups = [root.attachNewNode(name) for name in ('paint', 'glass', 'trim')]
    for group, metallic, roughness in zip(groups, (.48,.35,.18), (.24,.12,.44)):
        group.setMaterial(paint_material((1,1,1,1), metallic, roughness), 20)
    return groups


def build_detailed(parent, model, color):
    if model=='vf3':
        from src.visualization.vf3_mesh import build_vf3
        return build_vf3(parent,color)
    root = parent.attachNewNode('race_custom_body')
    root.setPythonTag('studio_mesh', True)
    paint, glass, trim = _groups(root)
    black, window = (.025,.031,.043,1), (.016,.031,.048,1)
    white, red = (.92,.98,1,1), (1,.035,.065,1)
    if model == 'lamborghini':
        loft(paint,'sculpted_body',[(-2.15,.73,-.14,.23),(-1.73,.92,-.16,.37),
            (-.75,.94,-.15,.33),(.55,.91,-.15,.31),(1.35,.86,-.13,.21),(2.10,.72,-.09,.07)],color)
        loft(glass,'canopy',[(-1.17,.61,.30,.39),(-.66,.62,.30,.85),
             (.02,.59,.28,.86),(.91,.64,.20,.27)],window)
        for side in (-1,1):
            strip(paint,'roof_rail',[(side*.60,-1.10,.41),(side*.62,-.65,.85),
                  (side*.59,.02,.86),(side*.64,.90,.27)],.045,color)
            panel(trim,'intake',[(side*.935,-.98,.02),(side*.95,-.42,.08),
                  (side*.94,-.53,.29),(side*.935,-1.10,.27)],black)
            strip(trim,'side_sill',[(side*.89,-1.85,-.12),(side*.95,-.6,-.13),
                  (side*.91,.70,-.12),(side*.82,1.70,-.08)],.065,black)
            # Angled triple LED, placed along the sloping front fascia.
            for dx,dy in ((-.10,0),(0,-.08),(.10,0)):
                strip(glass,'led',[(side*(.50+dx),1.78+dy,.20),
                      (side*(.55+dx),1.95+dy,.155)],.025,white)
            strip(glass,'rear_led',[(side*.27,-2.16,.23),(side*.67,-2.10,.25)],.035,red)
            loft(trim,'wing_stay',[(-1.82,.025,.29,.63),(-1.74,.025,.29,.63)],black,False).setX(side*.56)
        loft(trim,'rear_wing',[(-1.98,.91,.62,.65),(-1.73,.91,.59,.64)],black,False)
        loft(trim,'front_splitter',[(1.75,.86,-.13,-.09),(2.17,.75,-.12,-.09)],black,False)
    elif model == 'tesla':
        loft(paint,'sculpted_body',[(-2.16,.70,-.13,.23),(-1.67,.87,-.15,.38),
             (-.55,.93,-.15,.37),(.58,.91,-.15,.36),(1.53,.85,-.12,.27),(2.13,.70,-.06,.14)],color)
        loft(glass,'panoramic_canopy',[(-1.47,.63,.32,.41),(-.84,.66,.34,.99),
             (-.22,.63,.32,1.10),(.30,.61,.30,1.04),(1.17,.67,.26,.35)],window)
        for side in (-1,1):
            strip(paint,'window_surround',[(side*.63,-1.42,.42),(side*.66,-.82,.98),
                  (side*.63,-.22,1.10),(side*.61,.30,1.04),(side*.67,1.14,.35)],.037,color)
            strip(trim,'b_pillar',[(side*.68,-.22,.37),(side*.63,-.22,1.09)],.04,black)
            strip(trim,'rocker',[(side*.86,-1.76,-.13),(side*.93,-.4,-.13),
                  (side*.89,1.44,-.10)],.045,black)
            for y in (-.78,.52):
                strip(trim,'flush_handle',[(side*.926,y,.29),(side*.926,y+.21,.29)],.018,black)
            strip(glass,'swept_headlight',[(side*.36,2.04,.19),(side*.65,1.91,.24),
                  (side*.72,1.72,.28)],.038,white)
            strip(glass,'rear_led',[(side*.30,-2.17,.27),(side*.64,-2.12,.30),
                  (side*.79,-1.91,.34)],.035,red)
        loft(trim,'lower_grille',[(2.01,.49,-.055,.025),(2.12,.46,-.045,.03)],black,False)
    else:
        raise ValueError('Unknown car model')
    # Rounded mirror pods and modest, consistent wheel-arch trim.
    for side in (-1,1):
        loft(paint,'mirror',[(.38,.08,.42,.49),(.51,.11,.41,.51),(.65,.07,.42,.49)],color).setX(side*.96)
        for y in (-1.4166,1.05234):
            points = [(side*.922,y+.362*math.cos(a),.113+.362*math.sin(a))
                      for a in [i*math.pi/12 for i in range(13)]]
            strip(trim,'wheel_arch',points,.035,black)
    return root


_WHEEL_CACHE = {}


def build_wheel(parent, radius=.313, width=.25, model='lamborghini'):
    if model=='vf3':
        radius=.375
        width=.28
    key = (radius,width,model)
    if key not in _WHEEL_CACHE:
        root = NodePath('race_wheel_visual')
        rubber, metal = root.attachNewNode('rubber'), root.attachNewNode('alloy')
        rubber.setMaterial(paint_material((1,1,1,1),.02,.65),20)
        metal.setMaterial(paint_material((1,1,1,1),.86,.20),20)
        if model=='vf3':
            metal.setMaterial(paint_material((1,1,1,1),.38,.44),20)
        segments = 28
        profile = ((-.50,.78),(-.50,.94),(-.37,.995),(.37,.995),
                   (.50,.94),(.50,.78),(.35,.72),(-.35,.72))
        vertices = [(x*width,r*radius*math.cos(i*math.tau/segments),
                     r*radius*math.sin(i*math.tau/segments))
                    for x,r in profile for i in range(segments)]
        faces=[]
        for j in range(len(profile)):
            for i in range(segments):
                a,b=j*segments+i,j*segments+(i+1)%segments
                c,d=((j+1)%len(profile))*segments+(i+1)%segments,((j+1)%len(profile))*segments+i
                faces.append((a,b,c,d))
        mesh(rubber,'rounded_tire',vertices,faces,(.025,.030,.039,1))
        spokes = 7 if model=='tesla' else (6 if model=='vf3' else 5)
        for side in (-1,1):
            x=side*width*.505
            ring=[(x,radius*.72*math.cos(i*math.tau/segments),radius*.72*math.sin(i*math.tau/segments)) for i in range(segments)]
            panel(rubber,'brake_disc',ring,(.075,.09,.12,1))
            # VF3 reference has black steel wheels rather than silver spokes.
            if model=='vf3':
                _steel_wheel_face(metal,x,side,radius,segments)
                continue
            # Two thin alloy rings leave the black pockets visible.
            for r0,r1 in ((.70,.76),(.19,.25)):
                verts=[(x+side*.004,radius*r*math.cos(i*math.tau/segments),radius*r*math.sin(i*math.tau/segments))
                       for r in (r0,r1) for i in range(segments)]
                fs=[(i,(i+1)%segments,(i+1)%segments+segments,i+segments) for i in range(segments)]
                mesh(metal,'rim',verts,fs,(.64,.72,.83,1))
            for i in range(spokes):
                a=i*math.tau/spokes
                pts=[(x+side*.009,radius*r*math.cos(a+d),radius*r*math.sin(a+d))
                     for r,d in ((.21,-.19),(.71,-.09),(.71,.10),(.21,.22))]
                panel(metal,'spoke',pts,(.69,.77,.88,1))
        root.flattenStrong()
        root.setPythonTag('model', model)
        _WHEEL_CACHE[key]=root
    return _WHEEL_CACHE[key].copyTo(parent)


def _steel_wheel_face(parent,x,side,radius,segments):
    dark=(.045,.052,.057,1)
    face=[(x+side*.006,radius*.74*math.cos(i*math.tau/segments),
           radius*.74*math.sin(i*math.tau/segments)) for i in range(segments)]
    panel(parent,'steel_disc',face,dark)
    for i in range(12):
        angle=i*math.tau/12
        cy,cz=radius*.56*math.cos(angle),radius*.56*math.sin(angle)
        hole=[(x+side*.009,cy+radius*.083*math.cos(j*math.tau/8),
               cz+radius*.083*math.sin(j*math.tau/8)) for j in range(8)]
        panel(parent,'steel_vent',hole,(.008,.010,.012,1))
