use ndarray::{Array1, Array2};

pub fn compute_cost_linear_regularization(
    x: &Array2<f64>,
    y: &Array1<f64>,
    w: &Array1<f64>,
    b: f64,
    lambda: f64,
) -> f64 {
    let cost = lab2_mvlr::compute_cost(x, y, w, b);
    let m = x.shape()[0] as f64;
    let reg_term = (w.mapv(|wi| wi.powi(2)).sum()) * (lambda / (2.0 * m));
    cost + reg_term
}

pub fn compute_cost_logistic_regularization(
    x: &Array2<f64>,
    y: &Array1<f64>,
    w: &Array1<f64>,
    b: f64,
    lambda: f64,
) -> f64 {
    let cost = lab2_sigmoid::compute_cost_logistic(x, y, w, b);
    let m = x.shape()[0] as f64;
    let reg_term = (w.mapv(|wi| wi.powi(2)).sum()) * (lambda / (2.0 * m));
    cost + reg_term
}

pub fn gradient_logistic_regularization(
    x: &Array2<f64>,
    y: &Array1<f64>,
    w: &Array1<f64>,
    b: f64,
    lambda: f64,
) -> (Array1<f64>, f64) {
    let (dw, db) = lab2_sigmoid::gradient_logistic(x, y, w, b);
    let m = x.shape()[0] as f64;
    let reg_dw = dw + &(w.mapv(|wi| wi * (lambda / m)));
    (reg_dw, db)
}

pub fn gradient_linear_regularization(
    x: &Array2<f64>,
    y: &Array1<f64>,
    w: &Array1<f64>,
    b: f64,
    lambda: f64,
) -> (Array1<f64>, f64) {
    let (db, dw) = lab2_mvlr::compute_gradient(x, y, w, b);
    let m = x.shape()[0] as f64;
    let reg_dw = dw + &(w.mapv(|wi| wi * (lambda / m)));
    (reg_dw, db)
}
