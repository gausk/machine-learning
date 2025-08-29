use burn::module::Module;
use burn::nn::{
    Linear, LinearConfig,
    loss::{MseLoss, Reduction},
};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::{relu, sigmoid, tanh};
use burn::tensor::backend::AutodiffBackend;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Serialize, Deserialize, Module)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    #[default]
    None,
}

#[derive(Module, Debug)]
pub struct Layer<B: Backend> {
    linear: Linear<B>,
    activation: Activation,
}

impl<B: Backend> Layer<B> {
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: Activation,
        device: &B::Device,
    ) -> Self {
        let linear = LinearConfig::new(input_size, output_size).init(device);
        Self { linear, activation }
    }

    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let output = self.linear.forward(input);
        match self.activation {
            Activation::ReLU => relu(output),
            Activation::Sigmoid => sigmoid(output),
            Activation::Tanh => tanh(output),
            Activation::None => output,
        }
    }
}

#[derive(Module, Debug)]
pub struct NeuralNetwork<B: Backend> {
    layers: Vec<Layer<B>>,
}

impl<B: Backend> NeuralNetwork<B> {
    pub fn new(layers: Vec<Layer<B>>) -> Self {
        Self { layers }
    }

    pub fn forward(&self, mut x: Tensor<B, 2>) -> Tensor<B, 2> {
        for layer in &self.layers {
            x = layer.forward(x);
        }
        x
    }

    pub fn parameters(&self) {
        for (i, layer) in self.layers.iter().enumerate() {
            println!(
                "Layer {}: Weights: {}, Bias: {}",
                i + 1,
                layer.linear.weight.to_data(),
                match &layer.linear.bias {
                    Some(b) => b.to_data().to_string(),
                    None => "None".to_string(),
                }
            );
        }
    }
}

// Separate training function that works with autodiff backend
pub fn train_model<B: AutodiffBackend>(
    mut model: NeuralNetwork<B>,
    x: Tensor<B, 2>,
    y: Tensor<B, 2>,
    epochs: usize,
    learning_rate: f64,
) -> NeuralNetwork<B> {
    let mut optimizer = AdamConfig::new().init();
    let loss_fn = MseLoss::new();

    println!("Starting training...");
    for epoch in 0..epochs {
        let preds = model.forward(x.clone());
        let loss = loss_fn.forward(preds, y.clone(), Reduction::Mean);

        if epoch % 1000 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, loss.clone().into_scalar());
        }

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(learning_rate, model, grads);
    }

    model
}
