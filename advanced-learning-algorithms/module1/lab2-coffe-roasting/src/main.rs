use burn::backend::Autodiff;
use burn::optim::AdamConfig;
use burn::prelude::Backend;
use burn::prelude::Tensor;
use burn_ndarray::NdArray;
use lab1_neuron_and_layers::{
    Activation, Layer, SampleData, TaskType, TrainingConfig, train_model,
};
use lab2_coffe_roasting::{normalize_inputs_with_stats, normalize_single_input};
use rand::{Rng, SeedableRng, rngs::StdRng};

/// Creates a coffee roasting dataset.
/// - roasting duration: 12-15 minutes is best
/// - temperature range: 175-260C is best
pub fn load_coffee_data(n: usize) -> Vec<SampleData> {
    let mut rng = StdRng::seed_from_u64(2);
    let mut dataset = Vec::with_capacity(n);

    for _ in 0..n {
        let mut t = rng.r#gen::<f32>(); // temperature raw [0,1]
        let mut d = rng.r#gen::<f32>(); // duration raw [0,1]

        d = d * 4.0 + 11.5; // roasting duration: 12-15
        t = t * (285.0 - 150.0) + 150.0; // temperature: 150-285

        // classification condition
        let y_line = -3.0 / (260.0 - 175.0) * t + 21.0;
        let label = if t > 175.0 && t < 260.0 && d > 12.0 && d < 15.0 && d <= y_line {
            1.0
        } else {
            0.0
        };

        dataset.push(SampleData {
            input: vec![t, d],
            target: vec![label],
        });
    }

    dataset
}

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
    let stats_json = serde_json::to_string(&norm_stats).unwrap();
    std::fs::create_dir_all("artifacts").ok();
    std::fs::write("artifacts/normalization_stats.json", stats_json)
        .expect("Failed to save normalization stats");

    let model = train_model(
        "artifacts",
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
