//! wgpu renderer that I keep copying between projects and modifying and that really should just be
//! its own library at this point but I'm not sure how a good API would look so I guess I'll keep
//! copying it around.

use std::rc::Rc;

use wgpu::*;
use winit::{dpi::PhysicalSize, event_loop::EventLoopWindowTarget, window::WindowBuilder};
use zaru_image::Gpu;

use crate::image::Resolution;

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
            .build(event_loop)?;
        Ok(Self {
            win: Rc::new(win),
            resolution,
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
            inner: gpu.device().create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d::default(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                format,
                view_formats: &[],
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
            self.inner = gpu.device().create_texture(&TextureDescriptor {
                label: Some(&self.label),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.format,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.size = size;
        }

        gpu.queue().write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.inner,
                mip_level: 0,
                origin: Origin3d::default(),
                aspect: wgpu::TextureAspect::All,
            },
            data,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(size.width * 4),
                rows_per_image: None,
            },
            size,
        );

        reallocated
    }
}

struct RenderPipelines {
    /// Renders a full-screen texture.
    textured_quad: RenderPipeline,
}

impl RenderPipelines {
    fn create(
        surface_format: TextureFormat,
        device: &Device,
        shader: &ShaderModule,
        shared_bind_group_layout: &BindGroupLayout,
    ) -> Self {
        let textured_quad = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("textured_quad"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[shared_bind_group_layout],
                    push_constant_ranges: &[],
                }),
            ),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: "vert",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: "frag",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    write_mask: ColorWrites::ALL,
                    blend: None,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
        });

        Self { textured_quad }
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
            layout: shared_bind_group_layout,
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
    gpu: &'static Gpu,
    surface: Option<Surface>,
    render_pipelines: RenderPipelines,

    texture: Texture,

    shared_bind_group_layout: BindGroupLayout,
    bind_groups: BindGroups,

    /// Surface must be destroyed before `Window`.
    window: Window,
}

impl Renderer {
    pub fn new(window: Window, gpu: &'static Gpu) -> anyhow::Result<Self> {
        let surface = unsafe { gpu.instance().create_surface(&*window.win)? };
        let shader = gpu.device().create_shader_module(ShaderModuleDescriptor {
            label: Some("fullscreen texture shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        Ok(Self::with_surface(window, gpu, shader, surface))
    }

    fn with_surface(
        window: Window,
        gpu: &'static Gpu,
        shader: ShaderModule,
        surface: Surface,
    ) -> Self {
        let surface_format = *surface
            .get_capabilities(gpu.adapter())
            .formats
            .get(0)
            .expect("adapter cannot render to window surface");

        let shared_bind_group_layout =
            gpu.device()
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
            gpu.device(),
            &shader,
            &shared_bind_group_layout,
        );

        let texture = Texture::empty(&gpu, "texture");
        let bind_groups = BindGroups::create(gpu.device(), &shared_bind_group_layout, &texture);

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
            .device()
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
            rpass.set_bind_group(0, &self.bind_groups.textured_quad, &[]);
            rpass.draw(0..3, 0..1);
        }

        self.gpu.queue().submit([encoder.finish()]);
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
                self.gpu.device(),
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
            .get_capabilities(self.gpu.adapter())
            .formats
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
            view_formats: Vec::new(),
        };

        self.surface().configure(self.gpu.device(), &config);
    }
}
