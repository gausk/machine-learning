use lab3_cost_function::compute_cost;
use ndarray::Array1;

fn compute_gradient(x: &Array1<f64>, y: &Array1<f64>, w: f64, b: f64) -> (f64, f64) {
    let m = x.len();
    let mut dj_dw = 0.0;
    let mut dj_db = 0.0;

    for i in 0..m {
        let f_wb = w * x[i] + b;
        let dj_dw_i = (f_wb - y[i]) * x[i];
        let dj_db_i = f_wb - y[i];
        dj_db += dj_db_i;
        dj_dw += dj_dw_i;
    }
    dj_dw /= m as f64;
    dj_db /= m as f64;

    (dj_dw, dj_db)
}

pub fn gradient_descent(
    x: &Array1<f64>,
    y: &Array1<f64>,
    w_in: f64,
    b_in: f64,
    alpha: f64,
    num_iters: usize,
) -> (f64, f64, Vec<f64>, Vec<(f64, f64)>) {
    let mut w = w_in;
    let mut b = b_in;
    let mut cost_history = Vec::new();
    let mut wb_history = Vec::new();

    for _ in 0..num_iters {
        wb_history.push((w, b));
        cost_history.push(compute_cost(x, y, w, b));
        let (dj_dw, dj_db) = compute_gradient(x, y, w, b);
        w -= alpha * dj_dw;
        b -= alpha * dj_db;
    }

    (w, b, cost_history, wb_history)
}
