use burn::backend::Autodiff;
use burn::optim::AdamConfig;
use burn::prelude::Backend;
use burn_ndarray::NdArray;
use lab1_neuron_and_layers::{Activation, Layer, TaskType, TrainingConfig, train_model};
use lab1_relu::plot_xy;
use lab2_softmax::{evaluate_correctness, make_blobs, my_softmax};

fn main() {
    let z: Vec<f64> = (1..=5).map(|v| v as f64).collect();
    let y = my_softmax(&z);
    println!("Z: {:?}", z);
    println!("Softmax: {:?}", y);
    plot_xy(&z, &y, "softmax.png", "Softmax Activation");
    println!("Plot saved at softmax.png");

    let centers = vec![
        vec![-5.0, 2.0],
        vec![-2.0, -2.0],
        vec![1.0, 2.0],
        vec![5.0, -2.0],
    ];
    let data = make_blobs(&centers, 2000, 1.0);

    type MyBackend = Autodiff<NdArray<f32>>;
    let device = <MyBackend as Backend>::Device::default();

    // Obvious flow
    let layers_with_softmax: Vec<Layer<MyBackend>> = vec![
        Layer::new(2, 25, Activation::ReLU, &device),
        Layer::new(25, 15, Activation::ReLU, &device),
        Layer::new(15, 4, Activation::Softmax, &device),
    ];

    let model = train_model(
        "artifacts/obvious-softmax",
        TrainingConfig::new(AdamConfig::new())
            .with_num_epochs(10)
            .with_learning_rate(0.001),
        layers_with_softmax,
        device,
        data.clone(),
        TaskType::MultiClassification(false),
    );
    println!("Evaluating obvious flow with softmax layer...");
    evaluate_correctness(&model, &data, &device);

    // Preferred flow
    let layers_with_linear: Vec<Layer<MyBackend>> = vec![
        Layer::new(2, 25, Activation::ReLU, &device),
        Layer::new(25, 15, Activation::ReLU, &device),
        Layer::new(15, 4, Activation::None, &device),
    ];

    let model = train_model(
        "artifacts/preferred-softmax",
        TrainingConfig::new(AdamConfig::new())
            .with_num_epochs(10)
            .with_learning_rate(0.001),
        layers_with_linear,
        device,
        data.clone(),
        TaskType::MultiClassification(true),
    );
    println!("Evaluating preferred flow with final linear layer and with logits...");
    evaluate_correctness(&model, &data, &device);
}
