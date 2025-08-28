use burn::tensor::Tensor;
use burn_ndarray::{NdArray, NdArrayDevice as Device};
use lab1_neuron_and_layers::{Activation, Layer, NeuralNetwork};

fn main() {
    println!("Welcome to Lab1 on Neuron and Layers");
    let device = Device::default();
    let x: Tensor<NdArray, 2> = Tensor::from_floats([[1.0], [2.0]], &device);
    println!("Input Tensor: {:?}", x);

    let layer_linear: Layer<NdArray> = Layer::new(1, 1, Activation::None, &device);
    let network = NeuralNetwork::new(vec![layer_linear]);
    let y_pred = network.forward(x);
    println!("Output Linear: {:?}", y_pred);

    //let layer_sigmoid: Layer<NdArray> = Layer::new(2, 2, Activation::None, &device);
}
