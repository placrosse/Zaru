@group(0) @binding(0)
var samp: sampler;

@group(1) @binding(0)
var texture: texture_2d<f32>;

struct Vertex {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position)
    position: vec4<f32>,
    @location(0)
    uv: vec2<f32>,
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    return VertexOutput(vec4(vertex.position, 0.0, 1.0), vertex.uv);
}

fn sample(uv: vec2<f32>) -> vec4<f32> {
    let samp = textureSample(texture, samp, uv);

    // Attempts to sample from outside the source image result in `Color::NONE`.
    // FIXME: Use `ClampToBorder` instead, when available (avoids the sample op).
    if (any(uv > vec2(1.0, 1.0)) || any(uv < vec2(0.0, 0.0))) {
        return vec4(0.0);
    } else {
        return samp;
    }
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return sample(in.uv);
}
