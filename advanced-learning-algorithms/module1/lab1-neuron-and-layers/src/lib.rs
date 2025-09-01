pub mod data;
pub mod model;
pub mod train;

pub use data::SampleData;
pub use model::{Activation, Layer};
pub use train::{TaskType, TrainingConfig, train_model};
