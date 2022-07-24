#version 430

layout(local_size_x = 8, local_size_y = 8) in;

layout(rgba8, set = 0, binding = 0) uniform image2D output_texture;

void main() {
    ivec2 output_size = imageSize(output_texture);

    if(gl_GlobalInvocationID.x >= output_size.x ||
        gl_GlobalInvocationID.y >= output_size.y) {
        return;
    }


    vec2 uv = vec2(gl_GlobalInvocationID.xy) / output_size;
    vec4 pixel = vec4(uv.x, 0.0, uv.y, 1.0);

    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    imageStore(output_texture, pixel_coords, pixel);
}
