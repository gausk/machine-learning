use burn::backend::Autodiff;
use burn::optim::AdamConfig;
use burn::prelude::Backend;
use burn_ndarray::NdArray;
use lab1_neuron_and_layers::{Activation, Layer, TaskType, TrainingConfig, train_model};
use lab2_softmax::evaluate_correctness;
use practice_w1_hdr::load_data;

fn main() {
    println!(
        "Welcome to Week 2 practice assignment on Neural Networks for Handwritten Digit Recognition using Multiclass"
    );

    let data = load_data(5000);
    type MyBackend = Autodiff<NdArray<f32>>;
    let device = <MyBackend as Backend>::Device::default();

    let layers: Vec<Layer<MyBackend>> = vec![
        Layer::new(400, 25, Activation::ReLU, &device),
        Layer::new(25, 15, Activation::ReLU, &device),
        Layer::new(15, 10, Activation::None, &device),
    ];

    let model = train_model(
        "artifacts",
        TrainingConfig::new(AdamConfig::new())
            .with_num_epochs(40)
            .with_learning_rate(0.01),
        layers,
        device,
        data.clone(),
        TaskType::MultiClassification(true),
    );
    println!("Evaluating model on multiclass...");
    evaluate_correctness(&model, &data, &device, 10);
}
