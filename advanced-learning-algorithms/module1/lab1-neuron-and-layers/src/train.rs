use crate::data::SampleData;
use crate::data::SimpleBatch;
use crate::data::SimpleBatcher;
use crate::model::{Layer, NeuralNetwork};
use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::InMemDataset;
use burn::optim::AdamConfig;
use burn::optim::decay::WeightDecayConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::ClassificationOutput;
use burn::train::TrainOutput;
use burn::train::ValidStep;
use burn::train::metric::{AccuracyMetric, LossMetric};
use burn::train::{LearnerBuilder, RegressionOutput, TrainStep};

#[derive(Debug, Clone)]
pub enum TaskType {
    Classification,
    Regression,
    MultiClassification(bool),
}

#[derive(Config)]
pub struct TrainingConfig {
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
    #[config(default = 0.0)]
    pub lambda: f32,
}

impl<B: AutodiffBackend> TrainStep<SimpleBatch<B>, RegressionOutput<B>> for NeuralNetwork<B> {
    fn step(&self, batch: SimpleBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_regression(batch.inputs, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<SimpleBatch<B>, RegressionOutput<B>> for NeuralNetwork<B> {
    fn step(&self, batch: SimpleBatch<B>) -> RegressionOutput<B> {
        self.forward_regression(batch.inputs, batch.targets)
    }
}

impl<B: AutodiffBackend> TrainStep<SimpleBatch<B>, ClassificationOutput<B>> for NeuralNetwork<B> {
    fn step(&self, batch: SimpleBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.inputs, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<SimpleBatch<B>, ClassificationOutput<B>> for NeuralNetwork<B> {
    fn step(&self, batch: SimpleBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.inputs, batch.targets)
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
    task_type: TaskType,
) -> NeuralNetwork<B> {
    train_model_with_test_data(
        artifact_dir,
        config,
        layers,
        device,
        data,
        Vec::new(),
        task_type,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn train_model_with_test_data<B: AutodiffBackend + Backend>(
    artifact_dir: &str,
    config: TrainingConfig,
    layers: Vec<Layer<B>>,
    device: B::Device,
    train_data: Vec<SampleData>,
    test_data: Vec<SampleData>,
    task_type: TaskType,
    verbose: bool,
) -> NeuralNetwork<B> {
    match task_type {
        TaskType::Classification => train_classification_model(
            artifact_dir,
            config,
            layers,
            device,
            train_data,
            test_data,
            false,
            false,
            verbose,
        ),
        TaskType::Regression => train_regression_model(
            artifact_dir,
            config,
            layers,
            device,
            train_data,
            test_data,
            verbose,
        ),
        TaskType::MultiClassification(with_logits) => train_classification_model(
            artifact_dir,
            config,
            layers,
            device,
            train_data,
            test_data,
            true,
            with_logits,
            verbose,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn train_classification_model<B: AutodiffBackend + Backend>(
    artifact_dir: &str,
    config: TrainingConfig,
    layers: Vec<Layer<B>>,
    device: B::Device,
    train_data: Vec<SampleData>,
    test_data: Vec<SampleData>,
    is_multi_classification: bool,
    with_logits: bool,
    verbose: bool,
) -> NeuralNetwork<B> {
    create_artifacts_directory(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Failed to save config");

    B::seed(config.seed);

    let train_loader = DataLoaderBuilder::new(SimpleBatcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .shuffle(config.seed)
        .build(InMemDataset::new(train_data));

    let test_loader = DataLoaderBuilder::new(SimpleBatcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .shuffle(config.seed)
        .build(InMemDataset::new(test_data));

    let mut model = NeuralNetwork::new(layers);
    model.with_muti_classification(is_multi_classification);
    model.with_logits(with_logits);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            model,
            config
                .optimizer
                .with_weight_decay(Some(WeightDecayConfig::new(config.lambda)))
                .init(),
            config.learning_rate,
        );

    let model_trained = learner.fit::<SimpleBatch<B>, SimpleBatch<B::InnerBackend>, ClassificationOutput<B>, ClassificationOutput<B::InnerBackend>>(train_loader, test_loader);

    if verbose {
        model_trained.parameters();
    }
    model_trained
        .clone()
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Failed to save model");

    model_trained
}

fn train_regression_model<B: AutodiffBackend + Backend>(
    artifact_dir: &str,
    config: TrainingConfig,
    layers: Vec<Layer<B>>,
    device: B::Device,
    train_data: Vec<SampleData>,
    test_data: Vec<SampleData>,
    verbose: bool,
) -> NeuralNetwork<B> {
    create_artifacts_directory(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Failed to save config");

    B::seed(config.seed);

    let train_loader = DataLoaderBuilder::new(SimpleBatcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .shuffle(config.seed)
        .build(InMemDataset::new(train_data));

    let test_loader = DataLoaderBuilder::new(SimpleBatcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .shuffle(config.seed)
        .build(InMemDataset::new(test_data));

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            NeuralNetwork::new(layers),
            config
                .optimizer
                .with_weight_decay(Some(WeightDecayConfig::new(config.lambda)))
                .init(),
            config.learning_rate,
        );

    let model_trained = learner.fit::<SimpleBatch<B>, SimpleBatch<B::InnerBackend>, RegressionOutput<B>, RegressionOutput<B::InnerBackend>>(train_loader, test_loader);

    if verbose {
        model_trained.parameters();
    }
    model_trained
        .clone()
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Failed to save model");

    model_trained
}
