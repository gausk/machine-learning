use burn::config::Config;
use burn::{data::dataloader::batcher::Batcher, prelude::*};

#[derive(Debug, Config)]
pub struct SampleData {
    pub input: Vec<f32>,
    pub target: Vec<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct SimpleBatcher;

#[derive(Debug, Clone)]
pub struct SimpleBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 2>,
}

impl<B: Backend> Batcher<B, SampleData, SimpleBatch<B>> for SimpleBatcher {
    fn batch(&self, items: Vec<SampleData>, device: &B::Device) -> SimpleBatch<B> {
        let input_tensors: Vec<Tensor<B, 1>> = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_data(TensorData::from(item.input.as_slice()), device))
            .collect();

        let target_tensors: Vec<Tensor<B, 1>> = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_data(TensorData::from(item.target.as_slice()), device))
            .collect();

        // Stack into 2D
        let inputs = Tensor::stack(input_tensors, 0);
        let targets = Tensor::stack(target_tensors, 0);

        SimpleBatch { inputs, targets }
    }
}
