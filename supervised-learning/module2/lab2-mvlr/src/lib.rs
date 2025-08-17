use ndarray::Array1;
use ndarray::Array2;
use ndarray::s;
use plotters::prelude::*;

pub fn compute_cost(x: &Array2<f64>, y: &Array1<f64>, w: &Array1<f64>, b: f64) -> f64 {
    let m = x.shape()[0];
    let mut cost = 0.0;
    for i in 0..m {
        let f_wb_i = x.slice(s![i, ..]).dot(w) + b;
        cost += (f_wb_i - y[i]).powi(2);
    }
    cost / (2.0 * m as f64)
}

pub fn compute_gradient(
    x: &Array2<f64>,
    y: &Array1<f64>,
    w: &Array1<f64>,
    b: f64,
) -> (f64, Array1<f64>) {
    let m = x.shape()[0];
    let n = x.shape()[1];
    let mut dj_dw = Array1::zeros(n);
    let mut dj_db = 0.0;

    for i in 0..m {
        let err = (x.slice(s![i, ..]).dot(w) + b) - y[i];
        for j in 0..n {
            dj_dw[j] += err * x[[i, j]];
        }
        dj_db += err;
    }
    dj_dw /= m as f64;
    dj_db /= m as f64;

    (dj_db, dj_dw)
}

pub fn gradient_descent(
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
        j_history.push(compute_cost(x, y, &w, b));

        let (dj_db, dj_dw) = compute_gradient(x, y, &w, b);
        w = &w - &(&dj_dw * alpha);
        b -= dj_db * alpha;

        if i % (num_iters / 10) == 0 {
            println!("Iteration {:4}: Cost {:.2}", i, j_history.last().unwrap());
        }
    }

    (w, b, j_history)
}

pub fn plot_cost_function_by_iterations(cost_hist: &[f64], jump: usize, path: &str) {
    let root = BitMapBackend::new(path, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Cost Function by Iterations", ("sans-serif", 30))
        .margin(30)
        .x_label_area_size(40)
        .y_label_area_size(70)
        .build_cartesian_2d(
            1f64..cost_hist.len() as f64,
            0.0..*cost_hist
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
        )
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("Iterations")
        .y_desc("Cost")
        .y_label_offset(30)
        .draw()
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            cost_hist
                .iter()
                .enumerate()
                .filter(|(i, _)| i % jump == 0)
                .map(|(i, &pt)| ((i + 1) as f64, pt)),
            &RED,
        ))
        .unwrap();
}
