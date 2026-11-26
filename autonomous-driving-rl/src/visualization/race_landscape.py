"""Render-only road presentation; frozen road geometry and physics stay intact."""

import math

from panda3d.core import ATSNone, SamplerState, Texture, Vec2
from types import MethodType
from src.visualization.race_showroom import studio_environment


def marking_pixel_widths(pixels_per_meter):
    """Preserve the original 2/3-pixel marks at the reference 22 px/metre."""
    return (max(1,round(2*pixels_per_meter/22)),
            max(1,round(3*pixels_per_meter/22)))


def _reference_markings(terrain, center_point):
    current_map=terrain.engine.current_map
    if current_map is None:
        return terrain._race_original_semantics(center_point)
    white,yellow=marking_pixel_widths(terrain._semantic_map_pixel_per_meter)
    layers=['lane','lane_line']
    if terrain.engine.global_config['show_crosswalk']:
        layers.append('crosswalk')
    return current_map.get_semantic_map(center_point,size=terrain._semantic_map_size,
        pixels_per_meter=terrain._semantic_map_pixel_per_meter,
        white_line_thickness=white,yellow_line_thickness=yellow,layer=layers)


def visual_map_layout(bounds, max_dimension=8192, target_pixels_per_meter=64):
    """Budget detail around the road, rather than empty kilometres of terrain."""
    x_min, x_max, y_min, y_max = bounds
    size = max(32, math.ceil(max(x_max-x_min, y_max-y_min) + 32))
    dimension = min(max_dimension, math.ceil(size * target_pixels_per_meter))
    # get_semantic_map truncates size*ppm; avoid losing a row to float rounding.
    ppm = math.nextafter(dimension / size, math.inf)
    return ((x_min+x_max)/2, (y_min+y_max)/2), size, ppm


def _update_visual_semantics(terrain, center_point):
    current_map = terrain.engine.current_map
    texture = terrain._race_visual_tex
    from src.visualization.vietnam_scenery import update_vietnam_scenery
    if update_vietnam_scenery(terrain):
        previous = getattr(terrain, '_race_markings', None)
        if previous is not None:
            previous.removeNode()
            terrain._race_markings = None
        return
    if current_map is None:
        # Reset without a map must not retain a previous road's materials.
        import numpy as np
        semantics = np.zeros((1, 1, 1), dtype=np.uint8)
        map_center, size, ppm = center_point, terrain._terrain_size, 1
    else:
        from src.visualization.race_graphics import get_graphics_profile
        target = 64 if get_graphics_profile().name == 'balanced' else 32
        max_dimension = min(8192, terrain.engine.win.getGsg().getMaxTextureDimension())
        map_center, size, ppm = visual_map_layout(
            current_map.road_network.get_bounding_box(), max_dimension, target)
        # Paint is vector geometry; rasterizing it here would leave stair steps
        # around the smooth ribbons and duplicate their broken-line pattern.
        layers = ['lane']
        if terrain.engine.global_config['show_crosswalk']:
            layers.append('crosswalk')
        semantics = current_map.get_semantic_map(map_center, size=size,
            pixels_per_meter=ppm, line_sample_interval=.25,
            layer=layers)
    height, width = semantics.shape[:2]
    texture.setup2dTexture(width, height, Texture.TUnsignedByte, Texture.FRed)
    texture.setAutoTextureScale(ATSNone)
    texture.setMinfilter(SamplerState.FTNearest)
    texture.setMagfilter(SamplerState.FTNearest)
    texture.setWrapU(SamplerState.WMBorderColor)
    texture.setWrapV(SamplerState.WMBorderColor)
    texture.setBorderColor((0, 0, 0, 0))
    texture.setRamImage(semantics)
    # Terrain is always centred at center_point, which need not be map_center.
    terrain.origin.setShaderInput('race_visual_semantics', texture)
    terrain.origin.setShaderInput('race_semantic_uv_scale', terrain._terrain_size / size)
    terrain.origin.setShaderInput('race_semantic_uv_offset', Vec2(
        .5 + (center_point[0]-map_center[0])/size,
        .5 + (center_point[1]-map_center[1])/size))
    terrain._race_visual_info = dict(size=[width, height], pixels_per_meter=ppm,
                                     center=list(map_center), extent_m=size)
    from src.visualization.race_road_markings import rebuild_markings
    rebuild_markings(terrain)


def _landscape_semantics(terrain, center_point):
    # Called by Terrain.reset before its first render, including map changes.
    semantics = _reference_markings(terrain, center_point)
    _update_visual_semantics(terrain, center_point)
    return semantics


def shade_vehicle(root, shader):
    from simplepbr import Pipeline
    environment=studio_environment()
    root.setShader(shader)
    root.setShaderInput('sh_coeffs',environment.sh_coefficients)
    root.setShaderInput('brdf_lut',Pipeline._BRDF_LUT)
    root.setShaderInput('filtered_env_map',environment.filtered_env_map)
    root.setShaderInput('max_reflection_lod',environment.filtered_env_map.num_loadable_ram_mipmap_images)
    root.setShaderInput('global_shadow_bias',.005)


def configure_landscape(env, preview):
    pssm=env.engine.pssm
    if pssm is not None:
        pssm.use_pssm=False
        env.engine.render.setShaderInput('use_pssm',False)
        pssm.buffer.setActive(False)
    terrain=env.engine.terrain
    env._race_surface_shader=preview.scene.getShader()
    env.engine._race_surface_shader=env._race_surface_shader
    # Keep the simulator's full-extent semantic/height data. Only RGB cameras
    # use the separate detailed material map, rebuilt before every reset render.
    if not hasattr(terrain,'_race_original_semantics'):
        terrain._race_original_semantics=terrain.get_terrain_semantics
        terrain._race_visual_tex = Texture('race_road_materials')
        terrain.get_terrain_semantics=MethodType(_landscape_semantics,terrain)
        semantics=terrain.get_terrain_semantics(terrain.position)
        terrain.semantic_tex.clearRamImage()
        terrain.semantic_tex.setRamImage(semantics)
    from metadrive.constants import Semantics
    from src.visualization.race_terrain_shader import terrain_render_state
    env.engine._race_terrain_render_state = terrain_render_state()
    env.engine.cam.node().setTagState(Semantics.TERRAIN.label, env.engine._race_terrain_render_state)
    for name in ('road_texture','road_texture_normal','grass_tex','rock_tex','rock_tex_2'):
        texture=getattr(terrain,name,None)
        if texture is not None:
            texture.setAnisotropicDegree(8)
            texture.setMinfilter(SamplerState.FTLinearMipmapLinear)
    env._race_surface_shader=preview.scene.getShader()
    for vehicle in env.agents.values():
        shade_vehicle(vehicle.origin,env._race_surface_shader)
    def update_camera(task):
        env.engine.render.setShaderInput('camera_world_position',
                                         env.engine.cam.getPos(env.engine.render))
        scenery=getattr(terrain,'_vietnam_scenery',None)
        if scenery:
            scenery.root.setShaderInput('vn_time',float(task.time % 10000))
        return task.cont
    env.engine.render.setShaderInput('camera_world_position',
                                     env.engine.cam.getPos(env.engine.render))
    env.engine.taskMgr.add(update_camera,'race_surface_camera',sort=48)
