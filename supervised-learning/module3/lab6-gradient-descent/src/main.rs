use lab2_sigmoid::gradient_logistic;
use lab2_sigmoid::train_logistic_regression;
use ndarray::array;

fn main() {
    let x = array![
        [0.5, 1.5],
        [1.0, 1.0],
        [1.5, 0.5],
        [3.0, 0.5],
        [2.0, 2.0],
        [1.0, 2.5]
    ];
    let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    let w_tmp = array![2.0, 3.0];
    let b_tmp = 1.0;

    let (dw, db) = gradient_logistic(&x, &y, &w_tmp, b_tmp);
    println!("Gradient dw: {}, db: {}", dw, db);

    let w_init = array![0.0, 0.0];
    let b_init = 0.0;
    let alpha = 0.1;
    let num_iters = 10000;
    let (w_final, b_final, _) =
        train_logistic_regression(&x, &y, &w_init, b_init, alpha, num_iters);
    println!("Trained weights: {}", w_final);
    println!("Trained bias: {}", b_final);
}
