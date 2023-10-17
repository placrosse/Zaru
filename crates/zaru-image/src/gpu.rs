use std::sync::{Arc, Mutex, OnceLock};

use anyhow::anyhow;
use bytemuck::NoUninit;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt as _},
    *,
};

use crate::{blend, draw::lines};

/// A handle to a GPU.
///
/// Zaru uses a global GPU handle that can be access with [`Gpu::get()`]. This is the primary way to
/// interact with this type, the remaining constructors are rarely needed.
#[allow(unused)]
pub struct Gpu {
    instance: Arc<Instance>,
    adapter: Arc<Adapter>,
    device: Arc<Device>,
    queue: Arc<Queue>,

    /// A [`BindGroupLayout`] describing a single non-storage texture bound at index 0.
    pub(crate) bgl_single_texture: BindGroupLayout,
    /// A [`BindGroupLayout`] describing a single write-only storage texture bound at index 0.
    pub(crate) bgl_single_storage_texture_rgba8unorm_srgb: BindGroupLayout,
    /// A [`BindGroupLayout`] describing a single sampler bound at index 0.
    pub(crate) bgl_single_sampler: BindGroupLayout,
    /// A [`BindGroup`] that matches `bgl_single_sampler` with a single linear texture sample.
    pub(crate) bg_linear_sampler: BindGroup,

    /// An empty [`RenderPipeline`] that serves only to clear the render target.
    pub(crate) clear_pipeline: RenderPipeline,
    pub(crate) blend_pipeline: RenderPipeline,
    pub(crate) line_pipe: RenderPipeline,
    pub(crate) epaint: Mutex<egui_wgpu::Renderer>,
    pub(crate) fonts: epaint::Fonts,
}

static INSTANCE: OnceLock<Gpu> = OnceLock::new();

impl Gpu {
    /// Returns a reference to the global GPU handle.
    ///
    /// If the global GPU handle hasn't been initialized yet, an appropriate default GPU will be
    /// opened. If this fails, this method will panic.
    pub fn get() -> &'static Gpu {
        INSTANCE.get_or_init(|| pollster::block_on(Self::open()).unwrap())
    }

    /// Sets the global GPU handle.
    ///
    /// # Panics
    ///
    /// This will panic if the global GPU handle has already been initialized by a previous call to
    /// [`Gpu::set`], or if [`Gpu::get`] has ever been called.
    pub fn set(gpu: Self) {
        let mut error = true;
        INSTANCE.get_or_init(|| {
            error = false;
            gpu
        });

        if error {
            panic!("global GPU handle was already set");
        }
    }

    /// Returns a [`bool`] indicating whether the global [`Gpu`] context has been configured.
    ///
    /// Note that this is a global flag that can be changed by any thread at any time, so it can
    /// only be relied on when no other threads are interfering.
    pub fn is_set() -> bool {
        INSTANCE.get().is_some()
    }

    /// Opens a suitable default GPU.
    pub async fn open() -> anyhow::Result<Self> {
        // The OpenGL backend panics spuriously, so don't enable it.
        let backends = Backends::PRIMARY;
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        log::info!("available graphics adapters:");
        for adapter in instance.enumerate_adapters(backends) {
            let info = adapter.get_info();
            log_adapter("-", &info);
        }

        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .ok_or_else(|| anyhow!("no graphics adapter found"))?;
        log_adapter("using", &adapter.get_info());

        log::debug!("adapter features: {:?}", adapter.features());
        let unsupported = Features::all() - adapter.features();
        log::debug!("unsupported features: {:?}", unsupported);

        log::debug!("adapter limits: {:?}", adapter.limits());

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: None,
                    features: Features::empty(),
                    // Make sure we use the texture resolution limits from the adapter, so we can
                    // support large images.
                    limits: Limits::downlevel_defaults().using_resolution(adapter.limits()),
                },
                None,
            )
            .await?;

        Ok(Self::from_wgpu(instance, adapter, device, queue))
    }

    /// Creates a [`Gpu`] handle from an existing [`wgpu::Device`] and [`wgpu::Queue`].
    ///
    /// [`Device`] and [`Queue`] can be passed wrapped in [`Arc`]s, which allows sharing them
    /// outside of the library.
    pub fn from_wgpu(
        instance: impl Into<Arc<Instance>>,
        adapter: impl Into<Arc<Adapter>>,
        device: impl Into<Arc<Device>>,
        queue: impl Into<Arc<Queue>>,
    ) -> Self {
        let instance = instance.into();
        let adapter = adapter.into();
        let device = device.into();
        let queue = queue.into();
        Self::from_wgpu_impl(instance, adapter, device, queue)
    }

    fn from_wgpu_impl(
        instance: Arc<Instance>,
        adapter: Arc<Adapter>,
        device: Arc<Device>,
        queue: Arc<Queue>,
    ) -> Self {
        let bgl_single_texture = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("bgl_single_texture"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let bgl_single_storage_texture_rgba8unorm_srgb =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("bgl_single_storage_texture_rgba8unorm_srgb"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8UnormSrgb,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                }],
            });
        let bgl_single_sampler = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("bgl_single_sampler"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            }],
        });
        let clear_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("clear_shader"),
            source: ShaderSource::Wgsl(
                r#"
@vertex fn vertex() -> @builtin(position) vec4<f32> { return vec4(0.0); }
@fragment fn fragment() -> @location(0) vec4<f32> { return vec4(0.0); }
                "#
                .into(),
            ),
        });

        Self {
            bg_linear_sampler: device.create_bind_group(&BindGroupDescriptor {
                label: Some("bg_linear_sampler"),
                layout: &bgl_single_sampler,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Sampler(&device.create_sampler(
                        &SamplerDescriptor {
                            label: Some("linear_sampler"),
                            mag_filter: FilterMode::Linear,
                            min_filter: FilterMode::Linear,
                            ..Default::default()
                        },
                    )),
                }],
            }),
            clear_pipeline: device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("clear_pipeline"),
                layout: None,
                vertex: VertexState {
                    module: &clear_shader,
                    entry_point: "vertex",
                    buffers: &[],
                },
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                fragment: Some(FragmentState {
                    module: &clear_shader,
                    entry_point: "fragment",
                    targets: &[Some(TextureFormat::Rgba8UnormSrgb.into())],
                }),
                multiview: None,
            }),
            blend_pipeline: blend::create_pipeline(&device),
            line_pipe: lines::create_pipeline(&device),
            bgl_single_texture,
            bgl_single_storage_texture_rgba8unorm_srgb,
            bgl_single_sampler,
            epaint: egui_wgpu::Renderer::new(&device, TextureFormat::Rgba8Unorm, None, 1).into(),
            fonts: epaint::Fonts::new(
                1.0,
                device.limits().max_texture_dimension_2d as usize,
                Default::default(),
            ),

            instance,
            adapter,
            device,
            queue,
        }
    }

    /// Returns a reference to the [`Instance`].
    #[inline]
    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }

    /// Returns a reference to the [`Adapter`].
    #[inline]
    pub fn adapter(&self) -> &Arc<Adapter> {
        &self.adapter
    }

    /// Returns a reference to the [`Device`].
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns a reference to the [`Queue`].
    #[inline]
    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }

    /// Creates a [`Buffer`] containing `contents`.
    pub(crate) fn vertex_buffer<T: NoUninit>(&self, contents: &[T]) -> Buffer {
        self.device().create_buffer_init(&BufferInitDescriptor {
            label: None,
            usage: BufferUsages::VERTEX,
            contents: bytemuck::cast_slice(contents),
        })
    }
}

fn log_adapter(prefix: &str, info: &AdapterInfo) {
    let backend = match info.backend {
        wgpu::Backend::Empty => "dummy",
        wgpu::Backend::Vulkan => "Vulkan",
        wgpu::Backend::Metal => "Metal",
        wgpu::Backend::Dx12 => "DX12",
        wgpu::Backend::Dx11 => "DX11",
        wgpu::Backend::Gl => "OpenGL",
        wgpu::Backend::BrowserWebGpu => "WebGPU",
    };
    let device_type = match info.device_type {
        wgpu::DeviceType::Other => "Unknown",
        wgpu::DeviceType::IntegratedGpu => "iGPU",
        wgpu::DeviceType::DiscreteGpu => "dGPU",
        wgpu::DeviceType::VirtualGpu => "vGPU",
        wgpu::DeviceType::Cpu => "CPU",
    };
    log::info!("{} [{}] [{}] {}", prefix, backend, device_type, info.name);
}
