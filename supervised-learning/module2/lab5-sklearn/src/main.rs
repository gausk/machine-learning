use lab3_fs_and_lr::load_data;
use lab4_fe_and_pr::plot_xy_actual_predicted;
use linfa::Dataset;
use linfa::prelude::*;
use linfa::traits::Fit;
use linfa_linear::LinearRegression;
use ndarray::Array1;
use std::path::Path;

fn calculate_mse(actual: &Array1<f64>, predicted: &Array1<f64>) -> f64 {
    let n = actual.len() as f64;
    actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum::<f64>()
        / n
}

fn main() {
    // Normal flow
    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/house.txt");
    let (x_train, y_train) = load_data(data_path.to_str().unwrap(), 4);

    // Create a proper Dataset combining features (x_train) and targets (y_train)
    let dataset = Dataset::new(x_train.clone(), y_train.clone());

    let model = LinearRegression::default().fit(&dataset).unwrap();

    let predictions = model.predict(&dataset);
    let x_indices = ndarray::Array1::from_iter((1..=dataset.nsamples()).map(|x| x as f64));
    plot_xy_actual_predicted(
        &x_indices,
        &y_train,
        &predictions,
        "plot_predictions.png",
        "Rust Linfa LinearRegression House Data",
    );

    let mse = calculate_mse(&y_train, &predictions);
    println!("Mean Squared Error: {:.6}", mse);
    println!("\nModel Parameters:");
    println!("Intercept:  {}", model.intercept());
    println!("Parameters: {}", model.params());

    // Normalized flow
    let dataset = Dataset::new(x_train, y_train.clone());

    // Normalize features (equivalent to StandardScaler in Python)
    let mean = dataset.records().mean_axis(ndarray::Axis(0)).unwrap();
    let std = dataset.records().std_axis(ndarray::Axis(0), 0.0);

    // Apply standardization
    let records = dataset.records() - &mean;
    let records = &records / &std;
    let dataset = Dataset::new(records, y_train.clone());

    let model = LinearRegression::default().fit(&dataset).unwrap();

    let predictions = model.predict(&dataset);
    plot_xy_actual_predicted(
        &x_indices,
        dataset.targets(),
        &predictions,
        "plot_predictions_normalized.png",
        "Rust Linfa LinearRegression (Normalized)",
    );

    // Calculate and print MSE
    let mse = calculate_mse(dataset.targets(), &predictions);

    println!("Normalized Mean Squared Error: {:.6}", mse);
    println!("\nModel Parameters after normalization:");
    println!("Intercept:  {}", model.intercept());
    println!("Coefficients: {}", model.params());
}
