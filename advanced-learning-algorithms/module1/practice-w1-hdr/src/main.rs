use burn::backend::Autodiff;
use burn::optim::AdamConfig;
use burn::prelude::Tensor;
use burn::prelude::{Backend, TensorData};
use burn_ndarray::NdArray;
use lab1_neuron_and_layers::{
    Activation, Layer, SampleData, TaskType, TrainingConfig, train_model,
};
use ndarray::Array2;
use ndarray_npy::read_npy;
use std::path::Path;

fn load_data() -> Vec<SampleData> {
    let x_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/X.npy");
    let y_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/y.npy");
    let x_train: Array2<f64> = read_npy(x_path).unwrap();
    let y_train: Array2<u8> = read_npy(y_path).unwrap();
    let n = 1000.min(x_train.nrows()).min(y_train.nrows());
    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
        let input = x_train
            .row(i)
            .iter()
            .map(|x| *x as f32)
            .collect::<Vec<f32>>();
        let target = y_train
            .row(i)
            .iter()
            .map(|x| *x as f32)
            .collect::<Vec<f32>>();
        samples.push(SampleData::new(input, target));
    }
    samples
}

fn main() {
    println!(
        "Welcome to Week 1 practice assignment on Neural Networks for Handwritten Digit Recognition"
    );
    let data = load_data();
    println!("Input shape: [{}, {}], ", data.len(), data[0].input.len());
    println!("Target shape: [{}, {}], ", data.len(), data[0].target.len());

    type MyBackend = Autodiff<NdArray<f32>>;
    let device = <MyBackend as Backend>::Device::default();

    let layers: Vec<Layer<MyBackend>> = vec![
        Layer::new(400, 25, Activation::Sigmoid, &device),
        Layer::new(25, 15, Activation::Sigmoid, &device),
        Layer::new(15, 1, Activation::Sigmoid, &device),
    ];

    let model = train_model(
        "artifacts",
        TrainingConfig::new(AdamConfig::new())
            .with_num_epochs(20)
            .with_learning_rate(0.001),
        layers,
        device,
        data.clone(),
        TaskType::Classification,
    );

    let flat: Vec<f32> = data
        .iter()
        .flat_map(|d| d.input.clone()) // flatten Vec<Vec<f32>> into Vec<f32>
        .collect();

    let data_size = data.len();
    let target_dim = data[0].input.len();

    let x_test =
        Tensor::<MyBackend, 2>::from_data(TensorData::new(flat, [data_size, target_dim]), &device);

    let predicted = model.forward(x_test).to_data();
    let y_predicted = predicted.as_slice::<f32>().unwrap();

    let mut correct = 0;
    for i in 0..data_size {
        let y = data[i].target[0];
        let y_pred = y_predicted[i];

        if (y_pred - y).abs() < 0.5 {
            correct += 1;
        }
    }

    println!("Correct predictions: {} / {}", correct, data_size);
    println!(
        "Accuracy: {:.2}%",
        (correct as f32 / data_size as f32) * 100.0
    );
}
