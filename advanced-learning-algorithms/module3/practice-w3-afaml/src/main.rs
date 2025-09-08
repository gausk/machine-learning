use burn::backend::Autodiff;
use burn::optim::AdamConfig;
use burn::prelude::Backend;
use burn_ndarray::NdArray;
use lab1_neuron_and_layers::train::train_model_with_test_data;
use lab1_neuron_and_layers::{Activation, Layer, TaskType, TrainingConfig};
use lab2_softmax::make_blobs;
use ndarray::array;
use practice_w3_afaml::{eval_cat_err, eval_mse};

fn main() {
    println!("Welcome to practice lab of Week 3");
    let y_hat = array![2.4, 4.2];
    let y_tmp = array![2.3, 4.1];
    let output = eval_mse(&y_hat, &y_tmp);
    println!("MSE: {}", output);

    let y_hat = array![1, 2, 0];
    let y_tmp = array![1, 2, 3];
    let outut = eval_cat_err(&y_hat, &y_tmp);
    println!("MSE: {}", outut);

    let centers = vec![
        vec![-5.0, 2.0],
        vec![-2.0, -2.0],
        vec![1.0, 2.0],
        vec![5.0, -2.0],
        vec![8.0, 8.0],
        vec![-8.0, -8.0],
    ];
    let train_data = make_blobs(&centers, 1000, 1.0);
    let test_data = make_blobs(&centers, 300, 1.0);

    type MyBackend = Autodiff<NdArray<f32>>;
    let device = <MyBackend as Backend>::Device::default();

    let layers: Vec<Layer<MyBackend>> = vec![
        Layer::new(2, 6, Activation::ReLU, &device),
        Layer::new(6, 6, Activation::None, &device),
    ];

    let lambdas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3];

    for lambda in lambdas {
        train_model_with_test_data(
            "artifacts",
            TrainingConfig::new(AdamConfig::new())
                .with_num_epochs(1000)
                .with_learning_rate(0.01)
                .with_lambda(lambda),
            layers.clone(),
            device,
            train_data.clone(),
            test_data.clone(),
            TaskType::MultiClassification(true),
            false,
        );
    }
}
