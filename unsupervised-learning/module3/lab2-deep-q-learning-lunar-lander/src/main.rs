use burn::backend::Autodiff;
use burn::prelude::Backend;
use burn_ndarray::NdArray;
use lab1_neuron_and_layers::model::NeuralNetwork;
use lab1_neuron_and_layers::{Activation, Layer};

fn main() {
    println!("Welcome to lab on Deep Q Learning Lunar Lander!");
    type MyBackend = Autodiff<NdArray<f32>>;
    let device = <MyBackend as Backend>::Device::default();

    let q_network_layers: Vec<Layer<MyBackend>> = vec![
        Layer::new(8, 64, Activation::ReLU, &device),
        Layer::new(64, 64, Activation::ReLU, &device),
        Layer::new(64, 4, Activation::None, &device),
    ];
    let _q_network = NeuralNetwork::new(q_network_layers);

    let target_q_network_layers: Vec<Layer<MyBackend>> = vec![
        Layer::new(17, 256, Activation::ReLU, &device),
        Layer::new(256, 128, Activation::ReLU, &device),
        Layer::new(128, 32, Activation::None, &device),
    ];
    let _target_q_network = NeuralNetwork::new(target_q_network_layers);
}
