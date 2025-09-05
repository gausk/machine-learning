use burn::backend::Autodiff;
use burn::optim::AdamConfig;
use burn::prelude::Tensor;
use burn::prelude::{Backend, TensorData};
use burn_ndarray::NdArray;
use lab1_neuron_and_layers::{Activation, Layer, TaskType, TrainingConfig, train_model};
use practice_w1_hdr::load_data;

fn main() {
    println!(
        "Welcome to Week 1 practice assignment on Neural Networks for Handwritten Digit Recognition"
    );
    let data = load_data(1000);
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
