use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{
    Linear, LinearConfig,
    loss::{BinaryCrossEntropyLossConfig, MseLoss, Reduction},
};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::{relu, sigmoid, softmax, tanh};
use burn::tensor::backend::AutodiffBackend;
use burn::train::{ClassificationOutput, RegressionOutput};
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Serialize, Deserialize, Module)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    #[default]
    None,
    Softmax,
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
            Activation::Softmax => softmax(output, 1),
        }
    }
}

#[derive(Module, Debug)]
pub struct NeuralNetwork<B: Backend> {
    layers: Vec<Layer<B>>,
    is_multi_classification: bool,
    with_logits: bool,
}

impl<B: Backend> NeuralNetwork<B> {
    pub fn new(layers: Vec<Layer<B>>) -> Self {
        Self {
            layers,
            is_multi_classification: false,
            with_logits: false,
        }
    }

    pub fn with_muti_classification(&mut self, is_multi_classification: bool) {
        self.is_multi_classification = is_multi_classification;
    }

    pub fn with_logits(&mut self, logits: bool) {
        self.with_logits = logits;
    }

    pub fn forward(&self, mut x: Tensor<B, 2>) -> Tensor<B, 2> {
        for layer in &self.layers {
            x = layer.forward(x);
        }
        x
    }

    pub fn forward_classification(
        &self,
        x: Tensor<B, 2>,
        targets: Tensor<B, 2>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(x);
        let targets_int = targets.clone().int().reshape([-1]);
        let loss = if self.is_multi_classification {
            CrossEntropyLossConfig::new()
                .with_logits(self.with_logits)
                .init(&output.device())
                .forward(output.clone(), targets_int.clone())
        } else {
            BinaryCrossEntropyLossConfig::new()
                .with_logits(self.with_logits)
                .init(&output.device())
                .forward(output.clone(), targets.int())
        };
        ClassificationOutput::new(loss, output, targets_int)
    }

    pub fn forward_regression(
        &self,
        x: Tensor<B, 2>,
        targets: Tensor<B, 2>,
    ) -> RegressionOutput<B> {
        let output = self.forward(x);
        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Reduction::Mean);
        RegressionOutput::new(loss, output, targets)
    }

    pub fn parameters(&self) {
        println!("Model Parameters:");
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
