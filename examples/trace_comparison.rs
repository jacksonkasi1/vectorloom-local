//! Compare the shipping tracer with a detail-preserving configuration.
use anyhow::Result;
use vtracer::{ColorImage, Config, FitMode, Hierarchical, Preset};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(args.len() == 3, "usage: trace_comparison INPUT OUTPUT_DIR");
    let bytes = std::fs::read(&args[1])?;
    let dir = std::path::Path::new(&args[2]);
    std::fs::create_dir_all(dir)?;
    std::fs::write(
        dir.join("current.svg"),
        vectorloom_local::vectorize(&bytes)?.svg,
    )?;
    let rgba = image::load_from_memory(&bytes)?.to_rgba8();
    let source = ColorImage {
        width: rgba.width() as usize,
        height: rgba.height() as usize,
        pixels: rgba.into_raw(),
    };
    let mut config = Config::from_preset(Preset::Poster);
    config.mode = FitMode::Spline;
    config.hierarchical = Hierarchical::Cutout;
    config.filter_speckle = 0;
    config.length_threshold = 1.0;
    config.simplify = Some(0.2);
    config.max_colors = Some(36);
    config.optimize = 2;
    std::fs::write(dir.join("detail.svg"), config.build()?.to_svg(&source)?)?;
    Ok(())
}
