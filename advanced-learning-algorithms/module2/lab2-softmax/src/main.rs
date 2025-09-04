use burn::backend::Autodiff;
use burn::optim::AdamConfig;
use burn::prelude::{Backend, Tensor, TensorData};
use burn_ndarray::NdArray;
use lab1_neuron_and_layers::{Activation, Layer, TaskType, TrainingConfig, train_model};
use lab1_relu::plot_xy;
use lab2_softmax::{make_blobs, my_softmax};

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

    let layers: Vec<Layer<MyBackend>> = vec![
        Layer::new(2, 25, Activation::Sigmoid, &device),
        Layer::new(25, 15, Activation::Sigmoid, &device),
        Layer::new(15, 4, Activation::None, &device),
    ];

    let model = train_model(
        "artifacts",
        TrainingConfig::new(AdamConfig::new())
            .with_num_epochs(20)
            .with_learning_rate(0.001),
        layers,
        device,
        data.clone(),
        TaskType::MultiClassification,
    );

    let flat: Vec<f32> = data.iter().flat_map(|d| d.input.clone()).collect();

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
