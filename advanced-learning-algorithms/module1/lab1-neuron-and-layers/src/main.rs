use burn::tensor::Tensor;
use burn::backend::Autodiff;
use burn_ndarray::{NdArray, NdArrayDevice as Device};
use lab1_neuron_and_layers::{Activation, Layer, NeuralNetwork, train_model};

fn main() {
    println!("Welcome to Lab1 on Neuron and Layers");
    
    // Use Autodiff wrapper for training
    type MyBackend = Autodiff<NdArray<f32>>;
    let device = Device::default();
    
    let x: Tensor<MyBackend, 2> = Tensor::from_floats([[1.0], [2.0]], &device);
    println!("Input Tensor: {:?}", x);

    // Create a simple linear layer
    let layer_linear: Layer<MyBackend> = Layer::new(1, 1, Activation::None, &device);
    let network = NeuralNetwork::new(vec![layer_linear]);
    
    // Target values
    let y = Tensor::from_floats([[300.0], [400.0]], &device);
    
    // Train the network
    let trained_network = train_model(network, x.clone(), y.clone(), 10000, 0.1);
    
    // Get predictions
    let y_pred = trained_network.forward(x.clone());
    println!("Output Linear: {:?}", y_pred);

    // Example with sigmoid layer
    println!("\n--- Sequential Model with Sigmoid ---");
    let layers = vec![
        Layer::new(1, 4, Activation::Sigmoid, &device),
        Layer::new(4, 1, Activation::None, &device),
    ];
    let sigmoid_network = NeuralNetwork::new(layers);
    
    let trained_sigmoid_network = train_model(sigmoid_network, x.clone(), y.clone(), 1000, 1e-2);
    let y_pred_sigmoid = trained_sigmoid_network.forward(x);
    println!("Output with Sigmoid: {:?}", y_pred_sigmoid);
}