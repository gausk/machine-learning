use crate::data::SampleData;
use crate::data::SimpleBatch;
use crate::data::SimpleBatcher;
use crate::model::{Layer, NeuralNetwork};
use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::InMemDataset;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::TrainOutput;
use burn::train::ValidStep;
use burn::train::metric::LossMetric;
use burn::train::{LearnerBuilder, RegressionOutput, TrainStep};

#[derive(Config)]
pub struct TrainingConfig {
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

impl<B: AutodiffBackend> TrainStep<SimpleBatch<B>, RegressionOutput<B>> for NeuralNetwork<B> {
    fn step(&self, batch: SimpleBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_regression(batch.inputs, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

// impl<B: AutodiffBackend> TrainStep<SimpleBatch<B>, ClassificationOutput<B>> for NeuralNetwork<B> {
//     fn step(&self, batch: SimpleBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
//         let item = self.forward_classification(batch.inputs, batch.targets.argmax(1).squeeze(1));
//         TrainOutput::new(self, item.loss.backward(), item)
//     }
// }

impl<B: Backend> ValidStep<SimpleBatch<B>, RegressionOutput<B>> for NeuralNetwork<B> {
    fn step(&self, batch: SimpleBatch<B>) -> RegressionOutput<B> {
        self.forward_regression(batch.inputs, batch.targets)
    }
}

fn create_artifacts_directory(path: &str) {
    std::fs::remove_dir_all(path).ok();
    std::fs::create_dir_all(path).unwrap();
}

pub fn train_model<B: AutodiffBackend + Backend>(
    artifact_dir: &str,
    config: TrainingConfig,
    layers: Vec<Layer<B>>,
    device: B::Device,
    data: Vec<SampleData>,
) {
    create_artifacts_directory(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Failed to save config");

    B::seed(config.seed);

    let train_loader = DataLoaderBuilder::new(SimpleBatcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .shuffle(config.seed)
        .build(InMemDataset::new(data));

    let test_loader = DataLoaderBuilder::new(SimpleBatcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .shuffle(config.seed)
        .build(InMemDataset::new(vec![]));

    let learner = LearnerBuilder::new(artifact_dir)
        //.metric_train_numeric(AccuracyMetric::new())
        //.metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            NeuralNetwork::new(layers),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(train_loader, test_loader);

    model_trained.parameters();
    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Failed to save model");
}
