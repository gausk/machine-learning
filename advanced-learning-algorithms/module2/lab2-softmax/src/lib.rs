use lab1_neuron_and_layers::SampleData;
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
