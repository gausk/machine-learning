use lab2_mvlr::{gradient_descent, plot_cost_function_by_iterations};
use lab3_fs_and_lr::{load_data, zscore_normalize_features};
use ndarray::Array1;
use ndarray::Axis;
use std::path::Path;

fn main() {
    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/house.txt");
    let (x_train, y_train) = load_data(data_path.to_str().unwrap(), 4);
    println!(
        "Loaded training data with {} samples and {} features.",
        x_train.shape()[0],
        x_train.shape()[1]
    );

    let initial_w = Array1::zeros(x_train.shape()[1]);
    let initial_b = 0.;
    let iterations = 10;

    // high alpha
    let mut alpha = 9.9e-7;
    let (w_final, b_final, j_hist) =
        gradient_descent(&x_train, &y_train, &initial_w, initial_b, alpha, iterations);
    println!(
        "Final w: {}, Final b: {} with alpha {:e}",
        w_final, b_final, alpha
    );
    plot_cost_function_by_iterations(&j_hist, 1, "cost-function-plot-high-alpha.png");

    // medium alpha
    alpha = 9e-7;
    let (w_final, b_final, j_hist) =
        gradient_descent(&x_train, &y_train, &initial_w, initial_b, alpha, iterations);
    println!(
        "Final w: {}, Final b: {} with alpha {:e}",
        w_final, b_final, alpha
    );
    plot_cost_function_by_iterations(&j_hist, 1, "cost-function-plot-medium-alpha.png");

    // small alpha
    alpha = 1e-7;
    let (w_final, b_final, j_hist) =
        gradient_descent(&x_train, &y_train, &initial_w, initial_b, alpha, iterations);
    println!(
        "Final w: {}, Final b: {} with alpha {:e}",
        w_final, b_final, alpha
    );
    plot_cost_function_by_iterations(&j_hist, 1, "cost-function-plot-small-alpha.png");

    let (x_norm, mu, sigma) = zscore_normalize_features(&x_train);
    println!("X_mu = {}, \nX_sigma = {}", mu, sigma);
    println!(
        "Peak to Peak range by column in Raw        X:{:?}",
        x_train
            .axis_iter(Axis(1))
            .map(|col| {
                let max = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let min = col.iter().cloned().fold(f64::INFINITY, f64::min);
                max - min
            })
            .collect::<Vec<_>>()
    );
    println!(
        "Peak to Peak range by column in Normalized X:{:?}",
        x_norm
            .axis_iter(Axis(1))
            .map(|col| {
                let max = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let min = col.iter().cloned().fold(f64::INFINITY, f64::min);
                max - min
            })
            .collect::<Vec<_>>()
    );

    let (w_norm, b_norm, hist) =
        gradient_descent(&x_norm, &y_train, &initial_w, initial_b, 1.0e-1, 1000);
    println!(
        "Final w: {}, Final b: {} with normalized features",
        w_norm, b_norm
    );

    plot_cost_function_by_iterations(&hist[..100], 1, "cost-function-plot-normalize.png");
}
