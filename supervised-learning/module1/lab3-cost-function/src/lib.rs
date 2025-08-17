use ndarray::Array1;

/// Computes the cost for linear regression
pub fn compute_cost(x_train: &Array1<f64>, y_train: &Array1<f64>, w: f64, b: f64) -> f64 {
    let m = x_train.len();
    let cost_sum: f64 = x_train
        .iter()
        .zip(y_train.iter())
        .map(|(&x, &y)| (w * x + b - y).powi(2))
        .sum();
    cost_sum / (2 * m) as f64
}
