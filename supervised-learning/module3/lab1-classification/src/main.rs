use lab4_fe_and_pr::plot_xy_actual_predicted;
use linfa::Dataset;
use linfa::prelude::*;
use linfa::traits::Fit;
use linfa_linear::LinearRegression;
use ndarray::array;

fn main() {
    // Normal flow

    let x_train = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
    let y_train = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

    // Create a proper Dataset combining features (x_train) and targets (y_train)
    let dataset = Dataset::new(x_train, y_train.clone());

    let model = LinearRegression::default().fit(&dataset).unwrap();

    let predictions = model.predict(&dataset);
    let x_indices = ndarray::Array1::from_iter((0..dataset.nsamples()).map(|x| x as f64));
    plot_xy_actual_predicted(
        &x_indices,
        &y_train,
        &predictions,
        "plot.png",
        "Classification using Linear Regression",
    );

    println!("\nModel Parameters:");
    println!("Intercept:  {}", model.intercept());
    println!("Parameters: {}", model.params());
}
