// Top-level Game State Machine controlling menu navigation, world loading, gameplay, and pause.

#[derive(Debug, Clone, PartialEq)]
pub enum AppState {
    MainMenu,
    WorldSelect,
    WorldCreate,
    Loading { progress: f32, message: String },
    InGame,
    Paused,
}
