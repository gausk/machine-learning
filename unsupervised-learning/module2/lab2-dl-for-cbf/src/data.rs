use burn::config::Config;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::{Backend, Tensor, TensorData};
use csv::ReaderBuilder;
use rand::prelude::SliceRandom;
use rand::thread_rng;
use std::path::Path;

#[derive(Debug, Config)]
pub struct CBFData {
    user_features: Vec<f32>,
    movie_features: Vec<f32>,
    rating: f32,
}

impl CBFData {
    pub fn shape(&self) -> [usize; 2] {
        [self.user_features.len(), self.movie_features.len()]
    }
}

#[derive(Debug, Clone, Default)]
pub struct CBFBatcher;

#[derive(Debug, Clone)]
pub struct CBFBatch<B: Backend> {
    pub user_inputs: Tensor<B, 2>,
    pub movie_inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 2>,
}

impl<B: Backend> Batcher<B, CBFData, CBFBatch<B>> for CBFBatcher {
    fn batch(&self, items: Vec<CBFData>, device: &B::Device) -> CBFBatch<B> {
        let user_tensors: Vec<Tensor<B, 1>> = items
            .iter()
            .map(|item| {
                Tensor::<B, 1>::from_data(TensorData::from(item.user_features.as_slice()), device)
            })
            .collect();

        let movie_tensors: Vec<Tensor<B, 1>> = items
            .iter()
            .map(|item| {
                Tensor::<B, 1>::from_data(TensorData::from(item.movie_features.as_slice()), device)
            })
            .collect();

        let target_tensors: Vec<Tensor<B, 1>> = items
            .iter()
            .map(|item| {
                Tensor::<B, 1>::from_data(TensorData::from([item.rating].as_slice()), device)
            })
            .collect();

        // Stack into 2D
        let user_inputs = Tensor::stack(user_tensors, 0);
        let movie_inputs = Tensor::stack(movie_tensors, 0);
        let targets = Tensor::stack(target_tensors, 0);
        CBFBatch {
            user_inputs,
            movie_inputs,
            targets,
        }
    }
}

fn read_csv_to_matrx(path: &Path) -> Vec<Vec<f32>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .unwrap();
    rdr.records()
        .into_iter()
        .map(|record| {
            record
                .unwrap()
                .iter()
                .map(|x| x.parse::<f32>().unwrap())
                .collect::<Vec<f32>>()
        })
        .collect()
}

pub fn load_data(percent: f64) -> (Vec<CBFData>, Vec<CBFData>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data");
    let movie_train = read_csv_to_matrx(&path.join("content_user_train.csv"));
    let user_train = read_csv_to_matrx(&path.join("content_user_train.csv"));

    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_path(path.join("content_y_train.csv"))
        .unwrap();
    let y_train: Vec<f32> = rdr
        .records()
        .into_iter()
        .map(|record| record.unwrap()[0].parse::<f32>().unwrap())
        .collect();

    let mut data: Vec<CBFData> = movie_train
        .into_iter()
        .zip(y_train.into_iter())
        .zip(user_train.into_iter())
        .map(|((movie_features, rating), user_features)| CBFData {
            movie_features,
            rating,
            user_features,
        })
        .collect();

    data.shuffle(&mut thread_rng());
    let train_size = (data.len() as f64 * percent) as usize;
    let train_data = data[0..train_size].to_vec();
    let test_data = data[train_size..].to_vec();
    (train_data, test_data)
}
