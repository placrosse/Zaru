//! Shader loading, compilation and hot reloading.

use std::{borrow::Cow, fs};

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
        Ok(Self {
            textured_quad: Shader::load(device, loader, "quad.vert", "tex.frag")?,
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
        file: &str,
        dest: &mut Vec<u32>,
    ) -> anyhow::Result<()> {
        assert!(dest.is_empty());

        log::trace!("loading {:?} shader {}", stage, file);
        let source = fs::read_to_string(format!("shaders/{}", file))
            .with_context(|| format!("failed to load shader {}", file))?;
        let module = self
            .parser
            .parse(&stage.into(), &source)
            .map_err(|errs| anyhow::anyhow!("{}", errs.iter().format("\n")))
            .with_context(|| format!("failed to parse shader {}", file))?;

        let info = self
            .validator
            .validate(&module)
            .with_context(|| format!("shader {} failed validation", file))?;

        self.writer.write(&module, &info, None, dest)?;

        // Also dump the compiled SPIR-V to disk to allow inspection and debugging.
        let spv_dest = format!("shaders/{}.spv", file);
        let bytes = dest
            .iter()
            .flat_map(|word| word.to_ne_bytes())
            .collect::<Vec<_>>();
        fs::write(spv_dest, bytes)?;

        Ok(())
    }
}

impl Shader {
    fn load(device: &Device, loader: &mut Loader, vert: &str, frag: &str) -> anyhow::Result<Self> {
        let mut vert_spv = Vec::with_capacity(32);
        let mut frag_spv = Vec::with_capacity(32);

        loader.load_glsl(ShaderStage::Vertex, vert, &mut vert_spv)?;
        loader.load_glsl(ShaderStage::Fragment, frag, &mut frag_spv)?;

        let vert = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some(vert),
            source: wgpu::ShaderSource::SpirV(Cow::Borrowed(&vert_spv)),
        });
        let frag = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some(frag),
            source: wgpu::ShaderSource::SpirV(Cow::Borrowed(&frag_spv)),
        });

        Ok(Self { vert, frag })
    }
}
