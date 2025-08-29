use burn::backend::Autodiff;
use burn::tensor::Tensor;
use burn_ndarray::{NdArray, NdArrayDevice as Device};
use lab1_neuron_and_layers::{Activation, Layer, NeuralNetwork, train_model};

fn main() {
    println!("Welcome to Lab1 on Neuron and Layers");
    println!("--- Single Neuron Linear Model ---");
    type MyBackend = Autodiff<NdArray<f32>>;
    let device = Device::default();

    let x: Tensor<MyBackend, 2> = Tensor::from_floats([[1.0], [2.0]], &device);
    println!("Input Data: {}", x.to_data());

    let layer_linear: Layer<MyBackend> = Layer::new(1, 1, Activation::None, &device);
    let network = NeuralNetwork::new(vec![layer_linear]);

    let y = Tensor::from_floats([[300.0], [500.0]], &device);
    let trained_network = train_model(network, x.clone(), y.clone(), 10000, 0.1);
    let y_pred = trained_network.forward(x);
    println!("Expected: {}", y.to_data());
    println!("Predicted: {}", y_pred.to_data());
    println!("Model Parameters:");
    trained_network.parameters();

    // Example with sigmoid layer
    println!("\n--- Single Neuron with Sigmoid ---");
    let x: Tensor<MyBackend, 2> =
        Tensor::from_floats([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], &device);
    println!("Input Data: {}", x.to_data());

    let y: Tensor<MyBackend, 2> =
        Tensor::from_floats([[0.0], [0.0], [0.0], [1.0], [2.0], [3.0]], &device);
    let layers = vec![Layer::new(1, 1, Activation::Sigmoid, &device)];
    let sigmoid_network = NeuralNetwork::new(layers);

    let trained_sigmoid_network = train_model(sigmoid_network, x.clone(), y.clone(), 10000, 1e-2);
    let y_pred_sigmoid = trained_sigmoid_network.forward(x);
    println!("Expected: {}", y.to_data());
    println!("Predicted: {:.2}", y_pred_sigmoid.to_data());
    println!("Model Parameters:");
    trained_sigmoid_network.parameters();
}
