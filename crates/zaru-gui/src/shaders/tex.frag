#version 450

layout(set = 0, binding = 0) uniform texture2D texture;
layout(set = 0, binding = 1) uniform sampler texSampler;

layout(location = 0) in vec2 inTex;

layout(location = 0) out vec4 fragColor;

void main() {
    fragColor = texture(sampler2D(texture, texSampler), inTex);
}
