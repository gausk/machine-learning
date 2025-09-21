use burn::backend::Autodiff;
use burn::optim::AdamConfig;
use burn::prelude::Backend;
use burn::tensor::Tensor;
use burn_ndarray::NdArray;
use lab1_neuron_and_layers::{Activation, Layer, TrainingConfig};
use lab2_dl_for_cbf::data::load_data;
use lab2_dl_for_cbf::train::train_cbf_model;

fn main() {
    println!("Welcome to Lab on Deep Learning for Content based filtering");
    let (train_data, test_data, validate_data) = load_data(0.8, 20);
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
    println!(
        "Validating data shape: [{} {:?}]",
        validate_data.len(),
        validate_data[0].shape()
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
            .with_num_epochs(50)
            .with_learning_rate(0.01),
        user_layers,
        movie_layers,
        device,
        train_data,
        test_data,
    );
    for data in validate_data {
        let user_input: Tensor<MyBackend, 1> =
            Tensor::from_floats(data.user_features.as_slice(), &device);
        let movie_input: Tensor<MyBackend, 1> =
            Tensor::from_floats(data.movie_features.as_slice(), &device);
        let predicted_rating =
            model.forward(user_input.reshape([1, 17]), movie_input.reshape([1, 17]));
        println!(
            "Actual rating: {} Predicted rating: {}",
            data.rating,
            predicted_rating.into_data().as_slice::<f32>().unwrap()[0]
        );
    }
}
