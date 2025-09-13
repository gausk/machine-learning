use ndarray::{Array1, Array2, Axis};
use std::f64::consts::PI;
use std::path::Path;

pub fn load_data(suffix: &str) -> (Array2<f64>, Array2<f64>, Array1<u8>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"));
    let x: Array2<f64> =
        ndarray_npy::read_npy(path.join(format!("../data/X_{suffix}.npy"))).unwrap();
    let x_val: Array2<f64> =
        ndarray_npy::read_npy(path.join(format!("../data/X_val_{suffix}.npy"))).unwrap();
    let y_val: Array1<u8> =
        ndarray_npy::read_npy(path.join(format!("../data/y_val_{suffix}.npy"))).unwrap();
    (x, x_val, y_val)
}

pub fn calculate_mean_and_variance(x: &Array2<f64>) -> (Array1<f64>, Array1<f64>) {
    let mu = x.mean_axis(Axis(0)).unwrap();
    let variance = x.var_axis(Axis(0), 0.0);
    (mu, variance)
}

/// Computes the probability density function of the examples X under
/// the multivariate gaussian distribution with parameters mu and var.
/// Features are assumed independent here.
pub fn multivariate_gaussian(x: &Array2<f64>, mu: &Array1<f64>, var: &Array1<f64>) -> Array1<f64> {
    let n = x.nrows();
    let coefficient = var.mapv(|v| 1.0 / (2.0 * PI * v).sqrt()).product();
    let mut pdf = Array1::<f64>::zeros(n);
    for (i, row) in x.outer_iter().enumerate() {
        let diff = &row - mu;
        let exp_term = (&diff * &diff / (var * 2.0)).mapv(|v| (-v).exp());
        pdf[i] = coefficient * exp_term.product();
    }
    pdf
}

/// Finds the best threshold to use for selecting outliers based on the results
/// from a validation set (p_val) and the ground truth (y_val)
/// Returns epsilon and F1
pub fn select_threshold(p_val: &Array1<f64>, y_val: &Array1<u8>) -> (f64, f64) {
    let p_val_min = p_val.iter().cloned().fold(f64::NAN, f64::min);
    let p_val_max = p_val.iter().cloned().fold(f64::NAN, f64::max);
    let mut best_f1 = 0f64;
    let mut best_epsilon = 0f64;
    let n = y_val.len();
    let total_positives = y_val.sum() as f64;
    for epsilon in (0..1000).map(|i| p_val_min + (i as f64 / 1000f64) * (p_val_max - p_val_min)) {
        let mut true_positives = 0f64;
        let mut false_positives = 0f64;
        for i in 0..n {
            if p_val[i] <= epsilon {
                if y_val[i] == 1 {
                    true_positives += 1f64;
                } else {
                    false_positives += 1f64;
                }
            }
        }
        let precision = true_positives / (true_positives + false_positives);
        let recall = true_positives / total_positives;
        let f1 = (2.0 * precision * recall) / (precision + recall);
        if f1 > best_f1 {
            best_f1 = f1;
            best_epsilon = epsilon;
        }
    }
    (best_epsilon, best_f1)
}
