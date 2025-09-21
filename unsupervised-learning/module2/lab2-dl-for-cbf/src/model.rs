use burn::nn::loss::{MseLoss, Reduction};
use burn::prelude::{Backend, Module, Tensor};
use burn::train::RegressionOutput;
use lab1_neuron_and_layers::model::NeuralNetwork;

#[derive(Module, Debug)]
pub struct CBFModel<B: Backend> {
    pub user_network: NeuralNetwork<B>,
    pub movie_network: NeuralNetwork<B>,
}

impl<B: Backend> CBFModel<B> {
    pub fn new(user_network: NeuralNetwork<B>, movie_network: NeuralNetwork<B>) -> Self {
        Self {
            user_network,
            movie_network,
        }
    }

    pub fn forward_regression(
        &self,
        user_input: Tensor<B, 2>,
        movie_input: Tensor<B, 2>,
        targets: Tensor<B, 2>,
    ) -> RegressionOutput<B> {
        let output = self.forward(user_input, movie_input);
        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Reduction::Mean);
        RegressionOutput::new(loss, output, targets)
    }

    pub fn forward(&self, user_input: Tensor<B, 2>, movie_input: Tensor<B, 2>) -> Tensor<B, 2> {
        let user_out = self.user_network.forward(user_input);
        let movie_out = self.movie_network.forward(movie_input);
        user_out * movie_out
    }

    pub fn parameters(&self) {
        println!("User Model Parameters:");
        println!("{:?}", self.user_network.parameters());
        println!("Movie Model Parameters:");
        println!("{:?}", self.movie_network.parameters());
    }
}
