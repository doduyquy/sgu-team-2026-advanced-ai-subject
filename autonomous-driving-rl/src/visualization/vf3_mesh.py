"""Original VF3 study based on the user's turquoise/black photo reference.

The short, upright two-door silhouette follows the VF3's proportions.
Coordinates fit the shared visual wheelbase; this is not manufacturer CAD.
"""

import colorsys
import math
from src.visualization.race_car_mesh import mesh, loft, panel, _groups
from src.visualization.race_style import paint_material


def _face(parent,name,x0,x1,z0,z1,y,color,bevel=.025):
    r=min(bevel,(x1-x0)/4,(z1-z0)/4)
    points=[(x0+r,y,z0),(x1-r,y,z0),(x1,y,z0+r),(x1,y,z1-r),
            (x1-r,y,z1),(x0+r,y,z1),(x0,y,z1-r),(x0,y,z0+r)]
    return panel(parent,name,points,color)


def _bar(parent,name,a,b,width,color):
    # Proper camera-facing ribbon, including vertical V/letter strokes.
    dx,dz=b[0]-a[0],b[2]-a[2]
    length=max(math.hypot(dx,dz),1e-8)
    ox,oz=-dz/length*width/2,dx/length*width/2
    return panel(parent,name,[(a[0]+ox,a[1],a[2]+oz),(a[0]-ox,a[1],a[2]-oz),
                              (b[0]-ox,b[1],b[2]-oz),(b[0]+ox,b[1],b[2]+oz)],color)


def _side_bar(parent,name,side,y0,z0,y1,z1,width,color,x=.976):
    dy,dz=y1-y0,z1-z0
    length=max(math.hypot(dy,dz),1e-8)
    oy,oz=-dz/length*width/2,dy/length*width/2
    return panel(parent,name,[(side*x,y0+oy,z0+oz),(side*x,y1+oy,z1+oz),
                              (side*x,y1-oy,z1-oz),(side*x,y0-oy,z0-oz)],color)


def build_vf3(parent,color):
    root=parent.attachNewNode('race_custom_body')
    root.setPythonTag('studio_mesh',True)
    root.setPythonTag('vf3_reference','turquoise_black_two_door')
    paint,glass,trim=_groups(root)
    hue,saturation,value=colorsys.rgb_to_hsv(*color[:3])
    body=(*colorsys.hsv_to_rgb((hue-.045)%1,max(.88,saturation),value*.52),1)
    black=(.006,.009,.012,1)
    rubber=(.018,.022,.027,1)
    window=(.012,.024,.032,1)
    chrome=(.52,.60,.64,1)
    white=(.78,.84,.83,1)

    # Much taller shoulders and bonnet than the former low, elongated body.
    # Cut the lower shell around each wheel instead of hiding tyres with a
    # closed box. The actual Bullet wheels and suspension are left in place.
    ys={-2.04,-1.86,-.72,.65,1.47,1.69}
    for center in (-1.4166,1.05234):
        ys.update(center+offset for offset in (-.44,-.4,-.3,-.15,0,.15,.3,.4,.44))
    body_sections=[]
    for y in sorted(v for v in ys if -2.04<=v<=1.69):
        lo=.015
        for center in (-1.4166,1.05234):
            d=abs(y-center)
            if d<.44:
                lo=max(lo,.113+math.sqrt(max(0,.44*.44-d*d)))
        w=.985 if -.9<y<.8 else (.96 if y<1.47 else .87)
        hi=.98 if y<1.47 else .88
        body_sections.append((y,w,lo,hi))
    loft(paint,'boxy_body',body_sections,body,False)
    loft(paint,'bonnet',[ (.63,.80,.94,1.035),(1.36,.82,.93,1.015),
                         (1.66,.75,.84,.94)],body,False)
    loft(glass,'upright_cabin',[(-1.93,.80,.91,1.49),(-1.74,.85,.94,1.63),
         (.32,.84,.95,1.61),(.86,.86,.91,1.13)],window,False)
    # Black roof, shallow edge bevel; one large front door and rear quarter.
    roof=loft(trim,'black_roof',[(-1.95,.77,1.49,1.56),(-1.73,.87,1.59,1.64),
         (.30,.86,1.58,1.63),(.47,.80,1.54,1.60)],black,False)
    roof.setMaterial(paint_material((1,1,1,1),.02,.62),20)

    for side in (-1,1):
        # Upright rear glass, a single B pillar and slanted windscreen pillars.
        _side_bar(trim,'rear_pillar',side,-1.85,.94,-1.75,1.61,.095,black,x=.854)
        _side_bar(trim,'b_pillar',side,-.71,.95,-.71,1.64,.07,black,x=.862)
        _side_bar(trim,'a_pillar',side,.83,.99,.37,1.61,.075,black,x=.856)
        _side_bar(paint,'window_sill',side,-1.78,.94,.80,.94,.065,body,x=.916)
        _side_bar(trim,'door_shutline',side,-.76,.10,-.76,.94,.012,black)
        _side_bar(trim,'one_door_handle',side,-.55,.84,-.29,.84,.045,black,x=.989)
        loft(trim,'mirror_pod',[(.56,.065,.98,1.08),(.68,.115,.97,1.10),
                               (.81,.07,.98,1.07)],black,False).setX(side*1.01)

        # Wide, squared arch mouldings. All of their faces batch as trim.
        for center in (-1.4166,1.05234):
            well=[(side*.965,center+.43*math.cos(i*math.pi/16),
                   .113+.43*math.sin(i*math.pi/16)) for i in range(17)]
            well.extend(((side*.965,center-.43,-.12),
                         (side*.965,center+.43,-.12)))
            panel(trim,'wheel_well',well,black)
            outer=[(-.54,-.19),(-.54,.17),(-.39,.43),(-.22,.51),
                   (.22,.51),(.39,.43),(.54,.17),(.54,-.19)]
            inner=[(-.425,-.19),(-.425,.12),(-.31,.34),(-.18,.405),
                   (.18,.405),(.31,.34),(.425,.12),(.425,-.19)]
            verts=[(side*x,center+y,.113+z) for x in (.988,1.055)
                   for shape in (outer,inner) for y,z in shape]
            faces=[]
            for i in range(7):
                faces.extend(((i,i+1,i+9,i+8),(i+16,i+24,i+25,i+17),
                              (i,i+16,i+17,i+1),(i+8,i+9,i+25,i+24)))
            mesh(trim,'wide_arch',verts,faces,rubber)
        _side_bar(trim,'side_cladding',side,-1.89,.10,1.46,.10,.16,rubber,x=1.005)

    # Flat black front fascia and rectangular lamps, unlike the old tiny LEDs.
    _face(trim,'front_fascia',-.84,.84,.65,.99,1.710,black,.07)
    for side in (-1,1):
        x=side*.62
        _face(glass,'lamp_outer',x-.16,x+.16,.725,.935,1.728,chrome,.035)
        _face(trim,'lamp_recess',x-.128,x+.128,.752,.91,1.734,black,.025)
        points=[(x+.073*math.cos(i*math.tau/24),1.742,.832+.073*math.sin(i*math.tau/24)) for i in range(24)]
        panel(glass,'halogen_lens',points,white)
        _bar(glass,'chrome_brow',(side*.20,1.75,.925),(side*.78,1.75,.925),.035,chrome)
        _bar(glass,'vinfast_v',(0,1.752,.790),(side*.19,1.752,.937),.036,chrome)

    loft(trim,'front_bumper',[(1.45,.91,.015,.45),(1.78,.86,.045,.47)],rubber,False)
    _face(trim,'lower_grille',-.54,.54,.17,.30,1.787,black,.025)
    _face(glass,'plate',-.235,.235,.335,.455,1.795,(.70,.76,.74,1),.012)
    # Small vector plate letters, sharing the existing trim material.
    letters=[((-.16,.425),(-.125,.36)),((-.125,.36),(-.09,.425)),
             ((-.055,.36),(-.055,.425)),((-.055,.425),(.005,.425)),
             ((-.055,.394),(-.005,.394)),((.045,.425),(.11,.425)),
             ((.11,.425),(.11,.36)),((.045,.394),(.11,.394)),((.045,.36),(.11,.36))]
    for a,b in letters:
        _bar(trim,'plate_letter',(-a[0],1.801,a[1]),(-b[0],1.801,b[1]),.010,black)
    _face(glass,'skid_plate',-.57,.57,.055,.145,1.790,(.34,.39,.42,1),.018)
    # Rear hatch, vertical tail lamps and substantial lower cladding.
    loft(trim,'rear_bumper',[(-2.14,.83,.025,.35),(-1.88,.95,.015,.36)],rubber,False)
    for side in (-1,1):
        _face(glass,'tail_lamp',side*.72-.07,side*.72+.07,.62,.97,-2.052,(.62,.025,.035,1),.025)
    root.setPythonTag('reference_dimensions_mm',(3190,1679,1652))
    return root
