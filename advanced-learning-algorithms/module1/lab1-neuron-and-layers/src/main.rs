use burn::backend::Autodiff;
use burn::optim::AdamConfig;
use burn::prelude::Backend;
use burn_ndarray::NdArray;
use lab1_neuron_and_layers::{
    Activation, Layer, SampleData, TaskType, TrainingConfig, train_model,
};

fn main() {
    println!("Welcome to Lab1 on Neuron and Layers");
    println!("--- Single Neuron Linear Model ---");
    type MyBackend = Autodiff<NdArray<f32>>;
    let device = <MyBackend as Backend>::Device::default();

    let layer_linear: Layer<MyBackend> = Layer::new(1, 1, Activation::None, &device);
    let data = vec![
        SampleData::new(vec![1.0], vec![300.0]),
        SampleData::new(vec![2.0], vec![500.0]),
    ];
    train_model(
        "artifacts/linear",
        TrainingConfig::new(AdamConfig::new())
            .with_num_epochs(10000)
            .with_learning_rate(0.1),
        vec![layer_linear],
        device,
        data,
        TaskType::Regression,
    );

    println!("--- Single Neuron Classification Model ---");
    let layer_sigmoid: Layer<MyBackend> = Layer::new(1, 1, Activation::Sigmoid, &device);
    let data = vec![
        SampleData::new(vec![1.0], vec![0.0]),
        SampleData::new(vec![2.0], vec![0.0]),
        SampleData::new(vec![3.0], vec![0.0]),
        SampleData::new(vec![4.0], vec![1.0]),
        SampleData::new(vec![5.0], vec![1.0]),
        SampleData::new(vec![6.0], vec![1.0]),
    ];
    train_model(
        "artifacts/sigmoid",
        TrainingConfig::new(AdamConfig::new())
            .with_num_epochs(1000)
            .with_learning_rate(0.01),
        vec![layer_sigmoid],
        device,
        data,
        TaskType::Classification,
    );
}
