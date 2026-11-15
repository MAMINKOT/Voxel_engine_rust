mod camera;
mod crafting;
mod engine;
mod inventory;
mod player;
mod render;
mod sim;
mod ui;
mod voxel;

fn main() {
    // Control verbosity with RUST_LOG, e.g. RUST_LOG=debug cargo run
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,voxel_engine=info"),
    )
    .init();

    log::info!("Voxel Industrial Engine starting up");

    pollster::block_on(engine::run());
}
