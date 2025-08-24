use lab2_sigmoid::{compute_cost_logistic, gradient_logistic, sigmoid, train_logistic_regression};
use lab9_regularization::{
    compute_cost_logistic_regularization, gradient_logistic_regularization,
    train_logistic_regression_regularization,
};
use ndarray::{array, s};
use practice_w3_logistic_regression::{load_data, map_feature, predict_logistic};
use std::path::Path;

fn main() {
    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/ex2data1.txt");
    let (x_train, y_train) = load_data(data_path.to_str().unwrap(), 2);
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

    println!(
        "sigmoid [-1, 0, 1, 2] = {}",
        sigmoid(&array![-1., 0., 1., 2.])
    );

    let init_w = array![0., 0.];
    let init_b = 0.;

    println!(
        "Initial w = {}, b = {}, cost = {:.3}",
        init_w,
        init_b,
        compute_cost_logistic(&x_train, &y_train, &init_w, init_b)
    );

    let test_w = array![0.2, 0.2];
    let test_b = -24.0;
    println!(
        "Test w = {}, b = {}, cost = {:.3}",
        test_w,
        test_b,
        compute_cost_logistic(&x_train, &y_train, &test_w, test_b)
    );

    let (dj_dw, dj_db) = gradient_logistic(&x_train, &y_train, &init_w, init_b);
    println!(
        "Gradient at initial w = {}, b = {} is dj_dw = {}, dj_db = {}",
        init_w, init_b, dj_dw, dj_db
    );

    let test_w = array![0.2, -0.5];
    let (dj_dw, dj_db) = gradient_logistic(&x_train, &y_train, &test_w, test_b);
    println!(
        "Gradient at test w = {}, b = {} is dj_dw = {}, dj_db = {}",
        test_w, test_b, dj_dw, dj_db
    );

    println!("Learning parameters using gradient descent:");
    let initial_w = array![-0.00082978, 0.00220324];
    let initial_b = -8.0;
    let alpha = 0.001;
    let num_iters = 10000;

    let (w_learned, b_learned, j_history) =
        train_logistic_regression(&x_train, &y_train, &initial_w, initial_b, alpha, num_iters);

    println!(
        "Learned w = {}, b = {}, last cost = {:.3}",
        w_learned,
        b_learned,
        j_history.last().unwrap()
    );

    let tmp_x = array![
        [-1.02817175, -1.57296862],
        [0.36540763, -2.8015387],
        [1.24481176, -1.2612069],
        [-0.1809609, -0.74937038]
    ];
    let tmp_b = 0.3;
    let tmp_w = array![1.62434536, -0.61175641];
    let preds = predict_logistic(&tmp_x, &tmp_w, tmp_b);
    println!(
        "For X = {}, b = {}, predictions are {}",
        tmp_x, tmp_b, preds
    );

    let train_preds = predict_logistic(&x_train, &w_learned, b_learned);
    let accuracy = train_preds
        .iter()
        .zip(y_train.iter())
        .filter(|(pred, actual)| (**pred - **actual).abs() < 1e-5)
        .count() as f64
        / y_train.len() as f64
        * 100.0;
    println!("Train accuracy = {:.2}%", accuracy);

    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/ex2data2.txt");
    let (x_train, y_train) = load_data(data_path.to_str().unwrap(), 2);
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

    let x_mapped = map_feature(
        &x_train.slice(s![.., 0]).to_owned(),
        &x_train.slice(s![.., 1]).to_owned(),
    );
    println!(
        "Mapped training data with {} samples and {} features.",
        x_mapped.shape()[0],
        x_mapped.shape()[1]
    );

    let initial_w = array![
        -0.082978,
        0.22032449,
        -0.49988563,
        -0.19766743,
        -0.35324411,
        -0.40766141,
        -0.31373979,
        -0.15443927,
        -0.10323253,
        0.03881673,
        -0.08080549,
        0.1852195,
        -0.29554775,
        0.37811744,
        -0.47261241,
        0.17046751,
        -0.0826952,
        0.05868983,
        -0.35961306,
        -0.30189851,
        0.30074457,
        0.46826158,
        -0.18657582,
        0.19232262,
        0.37638915,
        0.39460666,
        -0.41495579
    ];

    let initial_b = 0.5;
    let lambda = 0.5;

    let cost =
        compute_cost_logistic_regularization(&x_mapped, &y_train, &initial_w, initial_b, lambda);
    println!(
        "With lambda = {}, regularized logistic cost = {:.3}",
        lambda, cost
    );

    let (dj_dw, dj_db) =
        gradient_logistic_regularization(&x_mapped, &y_train, &initial_w, initial_b, lambda);
    println!(
        "Regularized gradient at initial w and b with lambda = {} is dj_dw = {}, dj_db = {}",
        lambda, dj_dw, dj_db
    );

    let initial_b = 1.0;
    let alpha = 0.01;
    let num_iters = 10000;
    let lambda = 0.01;
    let (w_learned, b_learned) = train_logistic_regression_regularization(
        &x_mapped, &y_train, &initial_w, initial_b, alpha, num_iters, lambda,
    );

    println!(
        "Learned w = {}, b = {} with lambda = {} using logistic regression with regularization",
        w_learned, b_learned, lambda
    );

    println!(
        "Final cost: {}",
        compute_cost_logistic_regularization(&x_mapped, &y_train, &w_learned, b_learned, lambda)
    );

    let train_preds = predict_logistic(&x_mapped, &w_learned, b_learned);
    let accuracy = train_preds
        .iter()
        .zip(y_train.iter())
        .filter(|(pred, actual)| (**pred - **actual).abs() < 1e-5)
        .count() as f64
        / y_train.len() as f64
        * 100.0;
    println!("Train accuracy with regularization = {:.2}%", accuracy);
}
