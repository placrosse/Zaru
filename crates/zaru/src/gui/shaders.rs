//! Shader loading, compilation and hot reloading.

use std::borrow::Cow;

use anyhow::Context;
use itertools::Itertools;
use naga::{
    back::spv,
    front::glsl,
    valid::{Capabilities, ValidationFlags, Validator},
    ShaderStage,
};
use wgpu::{Device, ShaderModule};

#[derive(Debug)]
pub struct Shaders {
    pub textured_quad: Shader,
}

#[derive(Debug)]
pub struct Shader {
    pub vert: ShaderModule,
    pub frag: ShaderModule,
}

impl Shaders {
    pub fn load(device: &Device) -> anyhow::Result<Self> {
        let loader = &mut Loader::new()?;

        macro_rules! shader {
            ($vert:literal, $frag:literal) => {{
                static VERT: &str = include_str!(concat!("shaders/", $vert));
                static FRAG: &str = include_str!(concat!("shaders/", $frag));
                Shader::load_source(device, loader, $vert, $frag, VERT, FRAG)?
            }};
        }

        Ok(Self {
            textured_quad: shader!("quad.vert", "tex.frag"),
        })
    }
}

struct Loader {
    parser: glsl::Parser,
    validator: Validator,
    writer: spv::Writer,
}

impl Loader {
    fn new() -> anyhow::Result<Self> {
        // The SPIR-V writer flips the Y axis by default, but we operate fully in wgpu space, so
        // this needs to be turned off.
        let mut options = spv::Options::default();
        options.flags &= !spv::WriterFlags::ADJUST_COORDINATE_SPACE;

        Ok(Self {
            parser: glsl::Parser::default(),
            validator: Validator::new(ValidationFlags::all(), Capabilities::empty()),
            writer: spv::Writer::new(&options)?,
        })
    }

    fn load_glsl(
        &mut self,
        stage: ShaderStage,
        source: &str,
        path: &str,
        dest: &mut Vec<u32>,
    ) -> anyhow::Result<()> {
        assert!(dest.is_empty());

        let module = self
            .parser
            .parse(&stage.into(), &source)
            .map_err(|errs| anyhow::anyhow!("{}", errs.iter().format("\n")))
            .with_context(|| format!("failed to parse shader {}", path))?;

        let info = self
            .validator
            .validate(&module)
            .with_context(|| format!("shader {} failed validation", path))?;

        self.writer.write(&module, &info, None, dest)?;

        Ok(())
    }
}

impl Shader {
    fn load_source(
        device: &Device,
        loader: &mut Loader,
        vert_name: &str,
        frag_name: &str,
        vert_source: &str,
        frag_source: &str,
    ) -> anyhow::Result<Self> {
        let mut vert_spv = Vec::with_capacity(32);
        let mut frag_spv = Vec::with_capacity(32);

        loader.load_glsl(ShaderStage::Vertex, &vert_source, vert_name, &mut vert_spv)?;
        loader.load_glsl(
            ShaderStage::Fragment,
            &frag_source,
            frag_name,
            &mut frag_spv,
        )?;

        let vert = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(vert_name),
            source: wgpu::ShaderSource::SpirV(Cow::Borrowed(&vert_spv)),
        });
        let frag = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(frag_name),
            source: wgpu::ShaderSource::SpirV(Cow::Borrowed(&frag_spv)),
        });

        Ok(Self { vert, frag })
    }
}
