use lab9_regularization::{
    compute_cost_linear_regularization, compute_cost_logistic_regularization,
    gradient_linear_regularization,
};
use ndarray::{Array1, array};

fn main() {
    let x = array![
        [
            4.17022005e-01,
            7.20324493e-01,
            1.14374817e-04,
            3.02332573e-01,
            1.46755891e-01,
            9.23385948e-02
        ],
        [
            1.86260211e-01,
            3.45560727e-01,
            3.96767474e-01,
            5.38816734e-01,
            4.19194514e-01,
            6.85219500e-01
        ],
        [
            2.04452250e-01,
            8.78117436e-01,
            2.73875932e-02,
            6.70467510e-01,
            4.17304802e-01,
            5.58689828e-01
        ],
        [
            1.40386939e-01,
            1.98101489e-01,
            8.00744569e-01,
            9.68261576e-01,
            3.13424178e-01,
            6.92322616e-01
        ],
        [
            8.76389152e-01,
            8.94606664e-01,
            8.50442114e-02,
            3.90547832e-02,
            1.69830420e-01,
            8.78142503e-01
        ]
    ];
    let y: Array1<f64> = Array1::from(vec![0., 1., 0., 1., 0.]);
    let w = array![
        -0.40165317,
        -0.07889237,
        0.45788953,
        0.03316528,
        0.19187711,
        -0.18448437
    ];
    let b: f64 = 0.5;
    let lambda: f64 = 0.7;

    let linear_cost = compute_cost_linear_regularization(&x, &y, &w, b, lambda);
    println!("Regularized cost: {}", linear_cost);

    let logistic_cost = compute_cost_logistic_regularization(&x, &y, &w, b, lambda);
    println!("Regularized logistic cost: {}", logistic_cost);

    let x = array![
        [4.17022005e-01, 7.20324493e-01, 1.14374817e-04],
        [3.02332573e-01, 1.46755891e-01, 9.23385948e-02],
        [1.86260211e-01, 3.45560727e-01, 3.96767474e-01],
        [5.38816734e-01, 4.19194514e-01, 6.85219500e-01],
        [2.04452250e-01, 8.78117436e-01, 2.73875932e-02],
    ];

    let w = array![0.67046751, 0.4173048, 0.55868983];

    let (dw_lin, db_lin) = gradient_linear_regularization(&x, &y, &w, b, lambda);
    println!("Regularized linear gradient dw: {}", dw_lin);
    println!("Regularized linear gradient db: {}", db_lin);

    let (dw_log, db_log) =
        lab9_regularization::gradient_logistic_regularization(&x, &y, &w, b, lambda);
    println!("Regularized logistic gradient dw: {}", dw_log);
    println!("Regularized logistic gradient db: {}", db_log);
}
