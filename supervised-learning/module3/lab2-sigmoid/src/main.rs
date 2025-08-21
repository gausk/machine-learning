use lab2_sigmoid::{plot_xy, sigmoid, train_logistic_regression};
use lab4_fe_and_pr::plot_xy_actual_predicted;
use ndarray::Array1;
use ndarray::array;

fn main() {
    let z_tmp = Array1::range(-10.0, 11.0, 1.0);
    let y = sigmoid(&z_tmp);
    println!("z: {:e} y: {:e}", z_tmp, y);

    plot_xy(&z_tmp, &y, "sigmoid_plot.png");

    let x_train = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]];
    let y_train = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

    let w_in = Array1::zeros(1);
    let b_in = 0.0;

    let (w, b, _) = train_logistic_regression(&x_train, &y_train, &w_in, b_in, 0.01, 100000);
    println!("Trained weights: {}", w);
    println!("Trained bias: {}", b);
    let predictions = sigmoid(&(x_train.dot(&w) + b));
    println!("Predictions: {}", predictions);
    let x_range = Array1::range(0.0, 6.0, 1.0);
    plot_xy_actual_predicted(
        &x_range,
        &y_train,
        &predictions,
        "logistic-regression.png",
        "Logistic Regression",
    );
}
