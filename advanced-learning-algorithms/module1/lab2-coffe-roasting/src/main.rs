use burn::backend::Autodiff;
use burn::optim::AdamConfig;
use burn::prelude::Backend;
use burn::prelude::Tensor;
use burn_ndarray::NdArray;
use lab1_neuron_and_layers::{Activation, Layer, TaskType, TrainingConfig, train_model};
use lab2_coffe_roasting::{load_coffee_data, normalize_inputs_with_stats, normalize_single_input};

fn main() {
    println!("Welcome to Lab2 on Coffe Roasting");
    type MyBackend = Autodiff<NdArray<f32>>;
    let device = <MyBackend as Backend>::Device::default();

    let layers: Vec<Layer<MyBackend>> = vec![
        Layer::new(2, 3, Activation::Sigmoid, &device),
        Layer::new(3, 1, Activation::Sigmoid, &device),
    ];
    let mut data = load_coffee_data(200000);
    let norm_stats = normalize_inputs_with_stats(&mut data);

    // Save normalization stats for later inference
    norm_stats.save_to_file("artifacts/norm_stats.json");

    let model = train_model(
        "artifacts/coffee-roasting",
        TrainingConfig::new(AdamConfig::new())
            .with_num_epochs(10)
            .with_learning_rate(0.01),
        layers,
        device,
        data,
        TaskType::Classification,
    );

    println!("--- Inference Examples ---");
    let mut raw_inputs = vec![vec![200.0, 13.9], vec![200.0, 17.0]];

    println!("Raw inputs: {:?}", raw_inputs);

    // Apply the same normalization used during training
    for input in raw_inputs.iter_mut() {
        normalize_single_input(input, &norm_stats);
    }

    println!("Normalized inputs: {:?}", raw_inputs);

    let input_tensor = Tensor::from_floats(
        [
            [raw_inputs[0][0], raw_inputs[0][1]],
            [raw_inputs[1][0], raw_inputs[1][1]],
        ],
        &device,
    );

    let predictions = model.forward(input_tensor);
    println!("Predictions: {}", predictions.to_data());

    let pred_data = predictions.to_data();
    let values = pred_data.as_slice::<f32>().unwrap();
    for (i, &prob) in values.iter().enumerate() {
        let class = if prob > 0.5 {
            "Good Coffee"
        } else {
            "Poor Coffee"
        };
        println!("Sample {}: probability={:.3} -> {}", i, prob, class);
    }
}
