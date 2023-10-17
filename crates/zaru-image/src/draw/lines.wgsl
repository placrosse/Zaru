struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vertex(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    out.clip_position = vec4(vertex.position, 0.0, 1.0);
    out.color = vertex.color;

    return out;
}

@fragment
fn fragment(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
    return color;
}
