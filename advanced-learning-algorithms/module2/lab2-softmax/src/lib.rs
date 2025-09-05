use burn::prelude::{Backend, Tensor, TensorData};
use burn::tensor::Device;
use lab1_neuron_and_layers::SampleData;
use lab1_neuron_and_layers::model::NeuralNetwork;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

pub fn my_softmax(z: &[f64]) -> Vec<f64> {
    let exp_z = z.iter().map(|&x| x.exp()).collect::<Vec<_>>();
    let sum: f64 = exp_z.iter().sum();
    exp_z.into_iter().map(|exp_z| exp_z / sum).collect()
}

pub fn make_blobs(centers: &[Vec<f32>], n_samples: usize, std_dev: f32) -> Vec<SampleData> {
    let features = centers[0].len();
    let samples_per_center = n_samples / centers.len();
    let mut rng = thread_rng();

    let mut samples = Vec::with_capacity(n_samples);

    for (i, center) in centers.iter().enumerate() {
        let normal = Normal::new(0.0, std_dev).unwrap();
        for _ in 0..samples_per_center {
            let mut input = Vec::with_capacity(features);
            for &val in center.iter() {
                input.push(val + normal.sample(&mut rng));
            }
            samples.push(SampleData::new(input, vec![i as f32]))
        }
    }
    samples
}

pub fn evaluate_correctness<B: Backend>(
    model: &NeuralNetwork<B>,
    data: &[SampleData],
    device: &Device<B>,
) {
    let flat: Vec<f32> = data.iter().flat_map(|d| d.input.clone()).collect();

    let data_size = data.len();
    let target_dim = data[0].input.len();

    let x_test = Tensor::<B, 2>::from_data(TensorData::new(flat, [data_size, target_dim]), device);

    let predicted = model.forward(x_test).to_data();
    let y_predicted = predicted.as_slice::<f32>().unwrap();

    let mut correct = 0;
    for i in 0..data_size {
        let y = data[i].target[0] as usize;
        let mut max_prob = y_predicted[4 * i];
        let mut predicted_class = 0;
        for j in 1..4 {
            if y_predicted[4 * i + j] > max_prob {
                max_prob = y_predicted[4 * i + j];
                predicted_class = j;
            }
        }

        if predicted_class == y {
            correct += 1;
        }
    }

    println!("Correct predictions: {} / {}", correct, data_size);
    println!(
        "Accuracy: {:.2}%",
        (correct as f32 / data_size as f32) * 100.0
    );
}
