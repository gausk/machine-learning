use burn::backend::Autodiff;
use burn::optim::AdamConfig;
use burn::prelude::Backend;
use burn_ndarray::NdArray;
use lab1_neuron_and_layers::{Activation, Layer, TaskType, TrainingConfig, train_model};
use lab2_softmax::{evaluate_correctness, make_blobs};

fn main() {
    println!("Welcome to lab3 on MultiClass");
    let centers = vec![
        vec![-5.0, 2.0],
        vec![-2.0, -2.0],
        vec![1.0, 2.0],
        vec![5.0, -2.0],
    ];
    let data = make_blobs(&centers, 100, 1.0);

    type MyBackend = Autodiff<NdArray<f32>>;
    let device = <MyBackend as Backend>::Device::default();

    let layers: Vec<Layer<MyBackend>> = vec![
        Layer::new(2, 4, Activation::ReLU, &device),
        Layer::new(4, 4, Activation::None, &device),
    ];

    let model = train_model(
        "artifacts",
        TrainingConfig::new(AdamConfig::new())
            .with_num_epochs(200)
            .with_learning_rate(0.01),
        layers,
        device,
        data.clone(),
        TaskType::MultiClassification(true),
    );
    println!("Evaluating model on multiclass...");
    evaluate_correctness(&model, &data, &device);
}
