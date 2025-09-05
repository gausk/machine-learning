use lab1_neuron_and_layers::SampleData;
use ndarray::Array2;
use ndarray_npy::read_npy;
use std::path::Path;

pub fn load_data(n: usize) -> Vec<SampleData> {
    let x_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/X.npy");
    let y_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/y.npy");
    let x_train: Array2<f64> = read_npy(x_path).unwrap();
    let y_train: Array2<u8> = read_npy(y_path).unwrap();
    let n = n.min(x_train.nrows()).min(y_train.nrows());
    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
        let input = x_train
            .row(i)
            .iter()
            .map(|x| *x as f32)
            .collect::<Vec<f32>>();
        let target = y_train
            .row(i)
            .iter()
            .map(|x| *x as f32)
            .collect::<Vec<f32>>();
        samples.push(SampleData::new(input, target));
    }
    samples
}
