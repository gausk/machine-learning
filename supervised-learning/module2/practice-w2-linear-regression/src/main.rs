use lab2_mvlr::{compute_cost, compute_gradient, gradient_descent};
use lab3_fs_and_lr::load_data;
use ndarray::{array, s};
use std::path::Path;

fn main() {
    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/ex1data1.txt");
    let (x_train, y_train) = load_data(data_path.to_str().unwrap(), 1);

    println!(
        "Loaded training data with {} samples and {} features.",
        x_train.shape()[0],
        x_train.shape()[1]
    );
    println!(
        "First 5 samples:\nX = {}, \ny = {}",
        x_train.slice(s![0..5, ..]),
        y_train.slice(s![0..5])
    );

    let initial_w = array![2.0];
    let initial_b = 1.0;
    let cost = compute_cost(&x_train, &y_train, &initial_w, initial_b);
    println!(
        "Initial w = {}, b = {}, cost = {:.3}",
        initial_w, initial_b, cost
    );

    let initial_w = array![0.0];
    let initial_b = 0.0;
    let (dj_db, dj_dw) = compute_gradient(&x_train, &y_train, &initial_w, initial_b);
    println!(
        "Gradient at initial w = {}, b = {} is dj_dw = {}, dj_db = {}",
        initial_w, initial_b, dj_dw, dj_db
    );

    let test_w = array![0.2];
    let test_b = 0.2;
    let (dj_db, dj_dw) = compute_gradient(&x_train, &y_train, &test_w, test_b);
    println!(
        "Gradient at test w = {}, b = {} is dj_dw = {}, dj_db = {}",
        test_w, test_b, dj_dw, dj_db
    );

    let alpha = 0.01;
    let iterations = 1500;
    let (w_final, b_final, _) =
        gradient_descent(&x_train, &y_train, &initial_w, initial_b, alpha, iterations);
    println!(
        "Final w: {}, Final b: {} with alpha {} and iterations {}",
        w_final, b_final, alpha, iterations
    );

    let x1 = array![3.5];
    let predict1 = x1.dot(&w_final) + b_final;
    println!(
        "For population = 35,000, we predict a profit of ${:.2}",
        predict1 * 10000.0
    );

    let x2 = array![7.0];
    let predict2 = x2.dot(&w_final) + b_final;
    println!(
        "For population = 70,000, we predict a profit of ${:.2}",
        predict2 * 10000.0
    );
}
