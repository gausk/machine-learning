use burn::backend::Autodiff;
use burn::optim::AdamConfig;
use burn::prelude::Backend;
use burn_ndarray::NdArray;
use lab1_neuron_and_layers::{Activation, Layer, TrainingConfig};
use lab2_dl_for_cbf::data::load_data;
use lab2_dl_for_cbf::train::train_cbf_model;

fn main() {
    println!("Welcome to Lab on Deep Learning for Content based filtering");
    let (train_data, test_data) = load_data(0.8);
    println!(
        "Training data shape: [{} {:?}]",
        train_data.len(),
        train_data[0].shape()
    );
    println!(
        "Testing data shape: [{} {:?}]",
        test_data.len(),
        test_data[0].shape()
    );

    type MyBackend = Autodiff<NdArray<f32>>;
    let device = <MyBackend as Backend>::Device::default();

    let user_layers: Vec<Layer<MyBackend>> = vec![
        Layer::new(17, 256, Activation::ReLU, &device),
        Layer::new(256, 128, Activation::ReLU, &device),
        Layer::new(128, 32, Activation::None, &device),
    ];

    let movie_layers: Vec<Layer<MyBackend>> = vec![
        Layer::new(17, 256, Activation::ReLU, &device),
        Layer::new(256, 128, Activation::ReLU, &device),
        Layer::new(128, 32, Activation::None, &device),
    ];

    let model = train_cbf_model(
        "artifacts",
        TrainingConfig::new(AdamConfig::new())
            .with_num_epochs(30)
            .with_learning_rate(0.01),
        user_layers,
        movie_layers,
        device,
        train_data,
        test_data,
    );
    println!("Training model: [{}]", model);
}
