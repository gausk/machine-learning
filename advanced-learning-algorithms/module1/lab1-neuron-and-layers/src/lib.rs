use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::Tensor;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
struct NeuralNetwork<B: Backend> {
    layers: Vec<Linear<B>>,
}

#[allow(dead_code)]
impl<B: Backend> NeuralNetwork<B> {
    fn new(layer_sizes: &[usize], device: &B::Device) -> Self {
        let layers = layer_sizes
            .windows(2)
            .map(|sizes| LinearConfig::new(sizes[0], sizes[1]).init(device))
            .collect();
        Self { layers }
    }

    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.layers
            .iter()
            .fold(input, |acc, layer| layer.forward(acc))
    }
}
