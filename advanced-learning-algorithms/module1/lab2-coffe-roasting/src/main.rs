use burn::backend::Autodiff;
use burn::optim::AdamConfig;
use burn::prelude::Backend;
use burn::prelude::Tensor;
use burn_ndarray::NdArray;
use lab1_neuron_and_layers::{Activation, Layer, SampleData, TrainingConfig, train_model};
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

        // scale features
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
    let data = load_coffee_data(200000);
    let model = train_model(
        "artifacts",
        TrainingConfig::new(AdamConfig::new())
            .with_num_epochs(10)
            .with_learning_rate(0.01),
        layers,
        device,
        data,
    );

    let predict = model.forward(Tensor::from_floats([[200.0, 13.9], [200.0, 17.0]], &device));
    println!("{}", predict.to_data());
}
