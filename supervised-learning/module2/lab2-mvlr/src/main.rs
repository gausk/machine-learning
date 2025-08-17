use lab2_mvlr::{
    compute_cost, compute_gradient, gradient_descent, plot_cost_function_by_iterations,
};
use ndarray::Array1;
use ndarray::array;
use ndarray::s;

fn predict(x: &Array1<f64>, w: &Array1<f64>, b: f64) -> f64 {
    x.dot(w) + b
}

fn main() {
    let x_train = array![
        [2104.0, 5.0, 1.0, 45.0],
        [1416.0, 3.0, 2.0, 40.0],
        [852.0, 2.0, 1.0, 35.0]
    ];
    let y_train = array![460.0, 232.0, 178.0];

    println!("x shape: {:?}", x_train.shape());
    println!("{x_train}");
    println!("y shape: {:?}", y_train.shape());
    println!("{y_train}");

    let b_init = 785.1811367994083;
    let w_init = array![0.39133535, 18.75376741, -53.36032453, -26.42131618];
    println!("w_init: {}, b_init: {}", w_init, b_init);

    let x_vec = x_train.slice(s![0, ..]);
    println!("x_vec shape: {:?}, x_vec value: {}", x_vec.shape(), x_vec);

    // make a prediction
    let f_wb = predict(&x_vec.to_owned(), &w_init, b_init);
    println!("prediction: {}", f_wb);

    let cost = compute_cost(&x_train, &y_train, &w_init, b_init);
    println!("Cost at optimal w : {:e}", cost);

    let (tmp_dj_db, tmp_dj_dw) = compute_gradient(&x_train, &y_train, &w_init, b_init);
    println!("dj_db at initial w,b: {:e}", tmp_dj_db);
    println!("dj_dw at initial w,b: {:e}", tmp_dj_dw);

    let initial_w = Array1::zeros(w_init.len());
    let initial_b = 0.;
    let iterations = 1000;
    let alpha = 5.0e-7;

    // run gradient descent
    let (w_final, b_final, j_hist) =
        gradient_descent(&x_train, &y_train, &initial_w, initial_b, alpha, iterations);
    println!("b,w found by gradient descent: {:.2},{} ", b_final, w_final);

    let m = x_train.shape()[0];
    for i in 0..m {
        let x_row = x_train.slice(s![i, ..]).to_owned();
        let prediction = x_row.dot(&w_final) + b_final;
        println!(
            "prediction: {:.2}, target value: {}",
            prediction, y_train[i]
        );
    }

    // Plot the cost function
    plot_cost_function_by_iterations(&j_hist[..25], 1, "cost-function.png");
}
