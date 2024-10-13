mod engine;

use anyhow::Result;
use engine::Engine;

fn main() -> Result<()> {
    pretty_env_logger::init();

    let mut engine = Engine::new()?;
    engine.load_default_data()?;
    engine.run()?;

    Ok(())
}
