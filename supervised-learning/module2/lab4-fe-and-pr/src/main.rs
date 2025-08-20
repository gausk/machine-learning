use lab2_mvlr::gradient_descent;
use lab3_fs_and_lr::zscore_normalize_features;
use lab4_fe_and_pr::plot_xy_actual_predicted;
use ndarray::{Array1, Array2, Axis};

fn main() {
    let x: Array1<f64> = Array1::range(0.0, 20.0, 1.0);
    let y_train = x.mapv(|x| 1.0 + x.powi(2));
    let x_train = x.clone().insert_axis(Axis(1)).to_owned();
    println!("{}", y_train);

    let initial_w = Array1::zeros(x_train.shape()[1]);
    let initial_b = 0.;

    // using only x as feature
    let (model_w, model_b, _j_history) =
        gradient_descent(&x_train, &y_train, &initial_w, initial_b, 1e-2, 1000);
    println!("No feature engineering w: {}, b: {}", model_w, model_b);
    let y_predicted = x_train.dot(&model_w) + model_b;

    plot_xy_actual_predicted(
        &x,
        &y_train,
        &y_predicted,
        "plot_no_feature_engineering.png",
        "No feature Engineering",
    );

    // Feature engineering: create (20, 3) array with columns [x, x^2, x^3]
    let data: Vec<f64> = x
        .iter()
        .flat_map(|&v| vec![v, v.powi(2), v.powi(3)])
        .collect();

    let x_train_fe = Array2::from_shape_vec((x.len(), 3), data).unwrap();
    let initial_w = Array1::zeros(x_train_fe.shape()[1]);
    let (model_w_fe, model_b_fe, _j_history_fe) =
        gradient_descent(&x_train_fe, &y_train, &initial_w, initial_b, 1e-7, 10000);
    println!(
        "With feature engineering w: {}, b: {}",
        model_w_fe, model_b_fe
    );
    let y_predicted_fe = x_train_fe.dot(&model_w_fe) + model_b_fe;

    plot_xy_actual_predicted(
        &x,
        &y_train,
        &y_predicted_fe,
        "plot_with_feature_engineering.png",
        "Feature Engineering with x, x^2 and x^3",
    );

    // feature engineering and normalization
    let (x_norm_fe, mu, sigma) = zscore_normalize_features(&x_train_fe);
    println!(
        "Feature engineering and normalization mu: {}, sigma: {}",
        mu, sigma
    );

    let (model_w_fe, model_b_fe, _j_history_fe) =
        gradient_descent(&x_norm_fe, &y_train, &initial_w, initial_b, 1e-1, 10000);
    println!(
        "With feature engineering and normalization w: {}, b: {}",
        model_w_fe, model_b_fe
    );
    let y_predicted_fe = x_norm_fe.dot(&model_w_fe) + model_b_fe;

    plot_xy_actual_predicted(
        &x,
        &y_train,
        &y_predicted_fe,
        "plot_with_feature_engineering_and_normalization.png",
        "Feature Engineering and Normalization",
    );

    // complex function
    let y_complex = x.mapv(|x| (x / 2.0).cos());
    let data: Vec<f64> = x
        .iter()
        .flat_map(|&v| {
            vec![
                v,
                v.powi(2),
                v.powi(3),
                v.powi(4),
                v.powi(5),
                v.powi(6),
                v.powi(7),
                v.powi(8),
                v.powi(9),
            ]
        })
        .collect();

    let x_train_complex = Array2::from_shape_vec((x.len(), 9), data).unwrap();
    let (x_norm_complex_fe, mu, sigma) = zscore_normalize_features(&x_train_complex);
    println!("Complex feature normalization mu: {}, sigma: {}", mu, sigma);

    let initial_w = Array1::zeros(x_train_complex.shape()[1]);
    let (model_w_complex, model_b_complex, _j_history_complex) = gradient_descent(
        &x_norm_complex_fe,
        &y_complex,
        &initial_w,
        initial_b,
        1e-1,
        1000000,
    );
    println!(
        "With complex feature engineering w: {}, b: {}",
        model_w_complex, model_b_complex
    );
    let y_predicted_complex = x_norm_complex_fe.dot(&model_w_complex) + model_b_complex;

    plot_xy_actual_predicted(
        &x,
        &y_complex,
        &y_predicted_complex,
        "plot_with_complex_feature_engineering.png",
        "Feature Engineering for cos(x/2) function",
    );
}
