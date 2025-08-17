use lab4_gradient_descent::gradient_descent;
use ndarray::array;
use plotters::prelude::*;

fn plot_cost_function_by_iterations(cost_hist: &[f64], jump: usize, path: &str) {
    let root = BitMapBackend::new(path, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Cost Function by Iterations", ("sans-serif", 30))
        .margin(30)
        .x_label_area_size(40)
        .y_label_area_size(70)
        .build_cartesian_2d(
            0f64..cost_hist.len() as f64,
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
                .map(|(i, &pt)| (i as f64, pt)),
            &RED,
        ))
        .unwrap();
}

fn plot_w_by_iterations(w_hist: &[f64], jump: usize, path: &str) {
    let root = BitMapBackend::new(path, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("W by Iterations", ("sans-serif", 30))
        .margin(30)
        .x_label_area_size(40)
        .y_label_area_size(70)
        .build_cartesian_2d(0f64..w_hist.len() as f64, 0.0..200f64)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("Iterations")
        .y_desc("W")
        .y_label_offset(30)
        .draw()
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            w_hist
                .iter()
                .enumerate()
                .filter(|(i, _)| i % jump == 0)
                .map(|(i, &pt)| (i as f64, pt)),
            &RED,
        ))
        .unwrap();
}

fn main() {
    let x_train = array![1.0, 1.7, 2.0, 2.5, 3.0, 3.2];
    let y_train = array![250.0, 300.0, 480.0, 430.0, 630.0, 730.0];
    println!("x_train = {x_train}");
    println!("y_train = {y_train}");

    println!("x_train.shape: {}", x_train.dim());
    println!("Number of training examples: {}", x_train.len());

    let w_init = 0.0;
    let b_init = 0.0;
    let iterations = 1000;
    let tmp_alpha = 1.0e-2;
    let (w_final, b_final, cost_hist, wb_hist) =
        gradient_descent(&x_train, &y_train, w_init, b_init, tmp_alpha, iterations);
    println!("w_final = {w_final}");
    println!("b_final = {b_final}");

    plot_cost_function_by_iterations(
        &cost_hist[..100],
        1,
        "cost-function-by-iterations-first-100.png",
    );
    plot_cost_function_by_iterations(&cost_hist, 10, "cost-function-by-iterations-all.png");
    plot_w_by_iterations(
        &wb_hist
            .iter()
            .map(|(w, _)| *w)
            .take(100)
            .collect::<Vec<f64>>(),
        1,
        "w-by-iterations-first-100.png",
    );
}
