use crate::data::{CBFBatch, CBFBatcher, CBFData};
use crate::model::CBFModel;
use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::InMemDataset;
use burn::module::Module;
use burn::optim::decay::WeightDecayConfig;
use burn::prelude::Backend;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::LossMetric;
use burn::train::{LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep};
use lab1_neuron_and_layers::model::NeuralNetwork;
use lab1_neuron_and_layers::train::create_artifacts_directory;
use lab1_neuron_and_layers::{Layer, TrainingConfig};

impl<B: AutodiffBackend> TrainStep<CBFBatch<B>, RegressionOutput<B>> for CBFModel<B> {
    fn step(&self, batch: CBFBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_regression(batch.user_inputs, batch.movie_inputs, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<CBFBatch<B>, RegressionOutput<B>> for CBFModel<B> {
    fn step(&self, batch: CBFBatch<B>) -> RegressionOutput<B> {
        self.forward_regression(batch.user_inputs, batch.movie_inputs, batch.targets)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn train_cbf_model<B: AutodiffBackend + Backend>(
    artifact_dir: &str,
    config: TrainingConfig,
    user_layers: Vec<Layer<B>>,
    movie_layers: Vec<Layer<B>>,
    device: B::Device,
    train_data: Vec<CBFData>,
    test_data: Vec<CBFData>,
    verbose: bool,
) -> CBFModel<B> {
    create_artifacts_directory(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Failed to save config");

    B::seed(config.seed);

    let train_loader = DataLoaderBuilder::new(CBFBatcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .shuffle(config.seed)
        .build(InMemDataset::new(train_data));

    let test_loader = DataLoaderBuilder::new(CBFBatcher)
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
            CBFModel::new(
                NeuralNetwork::new(user_layers),
                NeuralNetwork::new(movie_layers),
            ),
            config
                .optimizer
                .with_weight_decay(Some(WeightDecayConfig::new(config.lambda)))
                .init(),
            config.learning_rate,
        );

    let model_trained = learner.fit::<CBFBatch<B>, CBFBatch<B::InnerBackend>, RegressionOutput<B>, RegressionOutput<B::InnerBackend>>(train_loader, test_loader);

    if verbose {
        model_trained.parameters();
    }
    model_trained
        .clone()
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Failed to save model");

    model_trained
}
