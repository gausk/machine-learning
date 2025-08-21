use ndarray::{Array1, Array2};
use plotters::prelude::*;

pub fn sigmoid_function(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| {
        let x_clamped = x.clamp(-500.0, 500.0);
        sigmoid_function(x_clamped)
    })
}
pub fn plot_xy(x: &Array1<f64>, y: &Array1<f64>, path: &str) {
    let root = BitMapBackend::new(path, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Sigmoid Function", ("sans-serif", 50))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(-10.0..10.0, 0.0..1.0)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("Input (z)")
        .y_desc("Output (y)")
        .draw()
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            x.iter().zip(y.iter()).map(|(&x, &y)| (x, y)),
            &RED,
        ))
        .unwrap();
}

pub fn compute_cost_logistic(x: &Array2<f64>, y: &Array1<f64>, w: &Array1<f64>, b: f64) -> f64 {
    let m = x.shape()[0] as f64;
    let predictions = sigmoid(&(x.dot(w) + b));
    let cost =
        -y.dot(&predictions.mapv(f64::ln)) - (1.0 - y).dot(&(1.0 - predictions).mapv(f64::ln));
    cost / m
}

pub fn gradient_logistic(
    x: &Array2<f64>,
    y: &Array1<f64>,
    w: &Array1<f64>,
    b: f64,
) -> (Array1<f64>, f64) {
    let m = x.shape()[0] as f64;
    let predictions = sigmoid(&(x.dot(w) + b));
    let error = predictions - y;

    let dw = x.t().dot(&error) / m;
    let db = error.sum() / m;

    (dw, db)
}

pub fn train_logistic_regression(
    x: &Array2<f64>,
    y: &Array1<f64>,
    w_in: &Array1<f64>,
    b_in: f64,
    alpha: f64,
    num_iters: usize,
) -> (Array1<f64>, f64, Vec<f64>) {
    let mut w = w_in.clone();
    let mut b = b_in;
    let mut j_history = Vec::new();

    for i in 0..num_iters {
        j_history.push(compute_cost_logistic(x, y, &w, b));

        let (dj_dw, dj_db) = gradient_logistic(x, y, &w, b);
        w = &w - &(&dj_dw * alpha);
        b -= dj_db * alpha;

        if i % (num_iters / 10) == 0 {
            println!("Iteration {:4}: Cost {:.2}", i, j_history.last().unwrap());
        }
    }

    (w, b, j_history)
}
