"""RGB-only terrain material filtering for the pinned MetaDrive renderer.

Semantic IDs cannot be interpolated as colors: mixing grass (0) and asphalt
(20) otherwise produces a white line (10). Decode each texel first, then mix
material coverage. The numeric map, height field and sensor shaders stay intact.
"""

from pathlib import Path


_SAMPLING = """
uniform sampler2D race_visual_semantics;
uniform float race_semantic_uv_scale;
uniform vec2 race_semantic_uv_offset;

vec4 race_decode(float value) {
    // Coverage order: white, asphalt, yellow, crosswalk. Ground is the remainder.
    if (value < 5.0) return vec4(0);
    if (value < 16.0) return vec4(1, 0, 0, 0);
    if (value < 26.0) return vec4(0, 1, 0, 0);
    if (value < 34.0) return vec4(0, 0, 1, 0);
    if (value >= 40.0 && value < 222.0) return vec4(0, 0, 0, 1);
    return vec4(0);
}

float race_texel(ivec2 pixel) {
    ivec2 size = textureSize(race_visual_semantics, 0);
    if (any(lessThan(pixel, ivec2(0))) || any(greaterThanEqual(pixel, size))) return 0.0;
    return texelFetch(race_visual_semantics, pixel, 0).r * 255.0;
}

vec4 race_coverage(vec2 uv) {
    vec2 pixel = uv * vec2(textureSize(race_visual_semantics, 0)) - 0.5;
    ivec2 base = ivec2(floor(pixel));
    vec2 fraction = fract(pixel);
    return mix(mix(race_decode(race_texel(base)),
                   race_decode(race_texel(base + ivec2(1, 0))), fraction.x),
               mix(race_decode(race_texel(base + ivec2(0, 1))),
                   race_decode(race_texel(base + ivec2(1, 1))), fraction.x), fraction.y);
}
"""

_MATERIALS = """
  // Smooth material coverage, never the numeric class IDs themselves.
  vec2 race_uv = (terrain_uv - vec2(0.5)) * race_semantic_uv_scale + race_semantic_uv_offset;
  vec4 coverage = race_coverage(race_uv);
  float paved = dot(coverage, vec4(1));
  float value = texture(race_visual_semantics, clamp(race_uv, 0.0, 1.0)).r * 255.0;

  float rock_2 = saturate(0.8 * (height - 0.07));
  rock_2 *= saturate(pow(saturate(1.0 - slope), 2.0)) * 2.0;
  rock_2 = saturate(rock_2);
  float rock = saturate(pow(slope, 1.2) * 15.0);
  float grass = saturate(1.0 - saturate(rock + rock_2));

  vec3 ground_diffuse = texture(grass_tex, terrain_uv * grass_tex_ratio).rgb * grass
      + texture(rock_tex, terrain_uv * rock_tex_ratio).rgb * rock
      + texture(rock_tex_2, terrain_uv * rock_tex_ratio_2).rgb * rock_2;
  vec3 ground_normal = (texture(grass_normal, terrain_uv * grass_tex_ratio).rgb * 2.0 - 1.0) * grass
      + (texture(rock_normal, terrain_uv * rock_tex_ratio).rgb * 2.0 - 1.0) * rock
      + (texture(rock_normal_2, terrain_uv * rock_tex_ratio_2).rgb * 2.0 - 1.0) * rock_2;

  float theta = (value - 40.0) * 2.0 / 180.0 * 3.1415926535;
  vec2 crosswalk_uv = mat2(cos(theta), sin(theta), -sin(theta), cos(theta)) * terrain_uv;
  vec3 diffuse = ground_diffuse * (1.0 - paved)
      + vec3(1.0) * coverage.x
      + texture(road_tex, terrain_uv * road_tex_ratio).rgb * coverage.y
      + vec3(1.0, 0.78, 0.0) * coverage.z
      + texture(crosswalk_tex, crosswalk_uv * road_tex_ratio).rgb * coverage.w;
  vec3 tex_normal_world = normalize(mix(normalize(tbn * ground_normal),
      get_normal(vec3(0), road_normal, road_tex_ratio, tbn), paved));
"""


def terrain_render_state():
    """Reuse pinned lighting/terrain tessellation, replacing RGB materials only."""
    import metadrive
    from panda3d.core import NodePath, Shader

    shaders = Path(metadrive.__file__).resolve().parent / 'shaders'
    vertex = (shaders / 'terrain.vert.glsl').read_text(encoding='utf-8')
    fragment = (shaders / 'terrain.frag.glsl').read_text(encoding='utf-8')
    start = fragment.index('  // get the color and terrain normal in world space')
    end = fragment.index('//   vec3 terrain_normal_view', start)
    fragment = fragment[:start] + _MATERIALS + '\n' + fragment[end:]
    fragment = fragment.replace('void main() {', _SAMPLING + '\nvoid main() {', 1)
    shader = Shader.make(Shader.SL_GLSL, vertex=vertex, fragment=fragment)
    node = NodePath('race_terrain_materials')
    node.setShader(shader)
    return node.getState()
