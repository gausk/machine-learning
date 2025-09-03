pub fn my_softmax(z: &[f64]) -> Vec<f64> {
    let exp_z = z.iter().map(|&x| x.exp()).collect::<Vec<_>>();
    let sum: f64 = exp_z.iter().sum();
    exp_z.into_iter().map(|exp_z| exp_z / sum).collect()
}

// fn make_blobs(centers: &[Vec<f64>], n_samples: usize, std_dev: f64)  {
//     let features = centers[0].len();
// }
