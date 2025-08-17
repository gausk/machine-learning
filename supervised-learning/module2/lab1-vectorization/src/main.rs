use ndarray::{Array, Array1};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::time::Instant;

fn my_dot_multiply(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let n = a.len();
    let mut sum = 0.0;
    for i in 0..n {
        sum += a[i] * b[i];
    }
    sum
}

fn my_dot_iter(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn main() {
    // matrix using ndarray
    let a = Array::random((3, 2), Uniform::new(0., 10.));
    println!("{:?}", a);

    let b = Array::random((10000000,), Uniform::new(0., 1.));
    let c = Array::random((10000000,), Uniform::new(0., 1.));

    let start = Instant::now();
    let dot = b.dot(&c);
    let duration_ndarray = start.elapsed();
    println!("ndarray time: {:?} dot product: {}", duration_ndarray, dot);

    let start = Instant::now();
    let dot = my_dot_multiply(&b, &c);
    let duration_my = start.elapsed();
    println!(
        "my implementation with multiply time: {:?} dot product: {}",
        duration_my, dot
    );

    let start = Instant::now();
    let dot = my_dot_iter(&b, &c);
    let duration_my = start.elapsed();
    println!(
        "my implementation with iter time: {:?} dot product: {}",
        duration_my, dot
    );
}
