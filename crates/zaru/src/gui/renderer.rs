//! wgpu renderer that I keep copying between projects and modifying and that really should just be
//! its own library at this point but I'm not sure how a good API would look so I guess I'll keep
//! copying it around.

use std::{iter, mem::size_of, num::NonZeroU32, rc::Rc};

use anyhow::anyhow;
use bytemuck::{Pod, Zeroable};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Adapter, Backends, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, Buffer, BufferAddress,
    BufferUsages, Color, ColorWrites, Device, Extent3d, Features, ImageDataLayout, Origin3d,
    PrimitiveTopology, Queue, RenderPipeline, SamplerBindingType, SamplerDescriptor, ShaderStages,
    Surface, TextureDescriptor, TextureFormat, TextureUsages, VertexAttribute, VertexBufferLayout,
    VertexFormat, VertexStepMode,
};
use winit::{dpi::PhysicalSize, event_loop::EventLoopWindowTarget, window::WindowBuilder};

use crate::image::Resolution;

use super::shaders::Shaders;

const BACKGROUND: Color = Color::BLACK;

#[derive(Clone)]
pub struct Window {
    pub(crate) win: Rc<winit::window::Window>,
    resolution: Resolution,
}

impl Window {
    pub fn open<T>(
        event_loop: &EventLoopWindowTarget<T>,
        title: &str,
        resolution: Resolution,
    ) -> anyhow::Result<Self> {
        let win = WindowBuilder::new()
            .with_resizable(false) // TODO make resizeable
            .with_inner_size(PhysicalSize::new(resolution.width(), resolution.height()))
            .with_title(title)
            .build(&event_loop)?;
        Ok(Self {
            win: Rc::new(win),
            resolution,
        })
    }
}

/// A handle to a GPU device and command queue.
pub struct Gpu {
    instance: wgpu::Instance,
    adapter: Adapter,
    device: Device,
    queue: Queue,
}

impl Gpu {
    pub async fn open() -> anyhow::Result<Self> {
        let backends = Backends::PRIMARY;
        let instance = wgpu::Instance::new(backends);

        log::info!("available graphics adapters:");
        for adapter in instance.enumerate_adapters(backends) {
            let info = adapter.get_info();
            log::info!("- {:?} {:?} {}", info.backend, info.device_type, info.name);
        }

        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .ok_or_else(|| anyhow!("no graphics adapter found"))?;
        let info = adapter.get_info();
        log::info!(
            "using {:?} {:?} {}",
            info.backend,
            info.device_type,
            info.name
        );

        log::debug!("adapter features: {:?}", adapter.features());
        let unsupported = Features::all() - adapter.features();
        log::debug!("unsupported features: {:?}", unsupported);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
                    limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
                },
                None,
            )
            .await?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }
}

struct Texture {
    inner: wgpu::Texture,
    size: Extent3d,
    label: String,
    format: TextureFormat,
}

impl Texture {
    fn empty(gpu: &Gpu, label: &str) -> Self {
        let format = TextureFormat::Rgba8UnormSrgb;
        Self {
            label: label.to_string(),
            inner: gpu.device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d::default(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                format,
            }),
            size: Extent3d::default(),
            format,
        }
    }

    fn update(&mut self, gpu: &Gpu, size: Extent3d, data: &[u8]) -> bool {
        assert_eq!((size.width * size.height * 4) as usize, data.len());

        let mut reallocated = false;

        // FIXME reuse the old texture if the new size is smaller
        if self.size != size {
            log::trace!(
                "reallocating texture '{}' ({}x{} -> {}x{})",
                self.label,
                self.size.width,
                self.size.height,
                size.width,
                size.height
            );
            reallocated = true;
            self.inner = gpu.device.create_texture(&TextureDescriptor {
                label: Some(&self.label),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.format,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            });
            self.size = size;
        }

        gpu.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.inner,
                mip_level: 0,
                origin: Origin3d::default(),
                aspect: wgpu::TextureAspect::All,
            },
            data,
            ImageDataLayout {
                offset: 0,
                // FIXME breaks with empty textures
                bytes_per_row: Some(NonZeroU32::new(size.width * 4).unwrap()),
                rows_per_image: None,
            },
            size,
        );

        reallocated
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 2],
    tex: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Instance {
    pos: [f32; 2],
    color: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SharedUniforms {
    view: [[f32; 4]; 3],
    quad_scale: f32,
    pad: [f32; 3],
}

struct RenderPipelines {
    /// Renders a textured `screen_space_quad`.
    textured_quad: RenderPipeline,
    /// A `[-1.0,-1.0]` to `[1.0,1.0]` 2D quad.
    screen_space_quad: Buffer,
}

impl RenderPipelines {
    fn create(
        surface_format: TextureFormat,
        device: &Device,
        shaders: &Shaders,
        shared_bind_group_layout: &BindGroupLayout,
    ) -> Self {
        fn v(x: f32, y: f32, u: f32, v: f32) -> Vertex {
            Vertex {
                pos: [x, y],
                tex: [u, v],
            }
        }
        let screen_space_quad = [
            v(-1.0, 1.0, 0.0, 0.0),  // top left
            v(1.0, 1.0, 1.0, 0.0),   // top right
            v(-1.0, -1.0, 0.0, 1.0), // bottom left
            v(1.0, -1.0, 1.0, 1.0),  // bottom right
        ];

        let screen_space_quad = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("screen_space_quad"),
            contents: bytemuck::bytes_of(&screen_space_quad),
            usage: BufferUsages::VERTEX,
        });

        let textured_quad = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("textured_quad"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&shared_bind_group_layout],
                    push_constant_ranges: &[],
                }),
            ),
            vertex: wgpu::VertexState {
                module: &shaders.textured_quad.vert,
                entry_point: "main",
                buffers: &[VertexBufferLayout {
                    array_stride: size_of::<Vertex>() as BufferAddress,
                    step_mode: VertexStepMode::Vertex,
                    attributes: &[
                        VertexAttribute {
                            format: VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32x2,
                            offset: 4 * 2,
                            shader_location: 1,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shaders.textured_quad.frag,
                entry_point: "main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    write_mask: ColorWrites::ALL,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
        });

        Self {
            textured_quad,
            screen_space_quad,
        }
    }
}

struct BindGroups {
    textured_quad: BindGroup,
}

impl BindGroups {
    fn create(
        device: &Device,
        shared_bind_group_layout: &BindGroupLayout,
        texture: &Texture,
    ) -> Self {
        let sampler = device.create_sampler(&SamplerDescriptor::default());
        let textured_quad = device.create_bind_group(&BindGroupDescriptor {
            label: Some("shared_bind_group"),
            layout: &shared_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(
                        &texture.inner.create_view(&Default::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
            ],
        });

        Self { textured_quad }
    }
}

pub struct Renderer {
    gpu: Rc<Gpu>,
    surface: Option<Surface>,
    render_pipelines: RenderPipelines,

    texture: Texture,

    shared_bind_group_layout: BindGroupLayout,
    bind_groups: BindGroups,

    /// Surface must be destroyed before `Window`.
    window: Window,
}

impl Renderer {
    pub fn new(window: Window, gpu: Rc<Gpu>) -> anyhow::Result<Self> {
        let surface = unsafe { gpu.instance.create_surface(&*window.win) };
        let shaders = Shaders::load(&gpu.device)?;
        Ok(Self::with_surface(window, gpu, shaders, surface))
    }

    fn with_surface(window: Window, gpu: Rc<Gpu>, shaders: Shaders, surface: Surface) -> Self {
        let surface_format = *surface
            .get_supported_formats(&gpu.adapter)
            .get(0)
            .expect("adapter cannot render to window surface");

        let shared_bind_group_layout =
            gpu.device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(SamplerBindingType::NonFiltering),
                            count: None,
                        },
                    ],
                });

        let render_pipelines = RenderPipelines::create(
            surface_format,
            &gpu.device,
            &shaders,
            &shared_bind_group_layout,
        );

        let texture = Texture::empty(&gpu, "texture");
        let bind_groups = BindGroups::create(&gpu.device, &shared_bind_group_layout, &texture);

        let mut this = Self {
            gpu,
            surface: Some(surface),
            render_pipelines,

            texture,

            shared_bind_group_layout,
            bind_groups,

            window,
        };
        this.recreate_swapchain();

        // Reloading shaders recreates the render pipelines, don't make that cause any panics.
        #[cfg(not(debug_assertions))]
        this.gpu.device.on_uncaptured_error(|err| {
            log::error!("wgpu error: {}\n", err);
        });

        this
    }

    fn surface(&self) -> &Surface {
        self.surface
            .as_ref()
            .expect("internal error: render surface is `None`")
    }

    pub fn redraw(&mut self) {
        let frame = match self.surface().get_current_texture() {
            Ok(frame) => frame,
            Err(err @ (wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost)) => {
                log::debug!("surface error: {}", err);
                self.recreate_swapchain();
                self.surface()
                    .get_current_texture()
                    .expect("failed to acquire next frame after recreating swapchain")
            }
            Err(e) => {
                panic!("failed to acquire frame: {}", e);
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let ops = wgpu::Operations {
                load: wgpu::LoadOp::Clear(BACKGROUND),
                store: true,
            };
            let color_attachment = wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops,
            };
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(color_attachment)],
                depth_stencil_attachment: None,
            });

            rpass.set_pipeline(&self.render_pipelines.textured_quad);
            rpass.set_vertex_buffer(0, self.render_pipelines.screen_space_quad.slice(..));
            rpass.set_bind_group(0, &self.bind_groups.textured_quad, &[]);
            rpass.draw(0..4, 0..1);
        }

        self.gpu.queue.submit(iter::once(encoder.finish()));
        frame.present();
    }

    pub fn update_texture(&mut self, res: Resolution, data: &[u8]) {
        let size = Extent3d {
            width: res.width(),
            height: res.height(),
            depth_or_array_layers: 1,
        };
        if self.texture.update(&self.gpu, size, data) {
            // When the texture is reallocated, the bind group containing it has to be recreated to
            // reflect that.
            self.bind_groups = BindGroups::create(
                &self.gpu.device,
                &self.shared_bind_group_layout,
                &self.texture,
            );
        }
    }

    pub fn window(&self) -> &winit::window::Window {
        &self.window.win
    }

    fn recreate_swapchain(&mut self) {
        let surface_format = *self
            .surface()
            .get_supported_formats(&self.gpu.adapter)
            .get(0)
            .expect("adapter cannot render to window surface");
        let res = self.window.win.inner_size();
        log::debug!(
            "creating target surface at {}x{} (format: {:?})",
            res.width,
            res.height,
            surface_format,
        );
        if res.width != self.window.resolution.width()
            || res.height != self.window.resolution.height()
        {
            // This should be impossible, since the window is not resizable.
            // Unfortunately, software.
            log::warn!(
                "window dimensions {}x{} do not match configured output resolution {}",
                res.width,
                res.height,
                self.window.resolution,
            );
        }
        let config = wgpu::SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: self.window.resolution.width(),
            height: self.window.resolution.height(),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
        };

        self.surface().configure(&self.gpu.device, &config);
    }
}
