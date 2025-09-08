use ndarray::Array1;

pub fn eval_mse(y: &Array1<f64>, yhat: &Array1<f64>) -> f64 {
    y.iter()
        .zip(yhat.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>()
        / (2.0 * yhat.len() as f64)
}

pub fn eval_cat_err(y: &Array1<i32>, yhat: &Array1<i32>) -> f64 {
    y.iter()
        .zip(yhat.iter())
        .map(|(&x1, &x2)| if x1 == x2 { 1.0 } else { 0.0 })
        .sum::<f64>()
        / (2.0 * yhat.len() as f64)
}
