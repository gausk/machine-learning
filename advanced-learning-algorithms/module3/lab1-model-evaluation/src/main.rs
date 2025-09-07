use lab1_model_evaluation::{linear_regression_training_with_polynomial, plot_mse};
use lab3_fs_and_lr::load_data;
use linfa::Dataset;
use linfa::dataset::Records;
use linfa::prelude::Fit;
use linfa::prelude::Predict;
use linfa::prelude::SingleTargetRegression;
use linfa::traits::Transformer;
use linfa_linear::LinearRegression;
use linfa_preprocessing::linear_scaling::LinearScaler;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::path::Path;

fn main() {
    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/data_w3_ex1.csv");
    let (x, y) = load_data(data_path.to_str().unwrap(), 1);

    let data = Dataset::new(x, y);
    println!("Input data shape: {:?}", data.nsamples());
    let mut rng = StdRng::seed_from_u64(42);
    let (data_train, data_test_and_validate) = data.shuffle(&mut rng).split_with_ratio(0.6);
    let (data_test, data_cv) = data_test_and_validate.split_with_ratio(0.5);
    println!("Train data shape: {:?}", data_train.nsamples());
    println!("Test data shape: {:?}", data_test.nsamples());
    println!("Cross validation data shape: {:?}", data_cv.nsamples());

    // Normalize input
    let scaler = LinearScaler::standard().fit(&data_train).unwrap();
    println!("Mean: {} Std: {}", scaler.offsets(), 1.0 / scaler.scales());
    let scaled_data_train = scaler.transform(data_train.clone());

    let model = LinearRegression::default().fit(&scaled_data_train).unwrap();

    let predict_train = model.predict(&scaled_data_train);

    let train_mse_error = predict_train
        .mean_squared_error(&scaled_data_train)
        .unwrap()
        / 2.0;
    println!("Training MSE: {}", train_mse_error);

    let scaled_data_cv = scaler.transform(data_cv.clone());
    let predict_cv = model.predict(&scaled_data_cv);
    let cv_mse_error = predict_cv.mean_squared_error(&scaled_data_cv).unwrap() / 2.0;
    println!("CV MSE: {}", cv_mse_error);

    let mut mse_trains = Vec::new();
    let mut mse_cvs = Vec::new();
    for degree in 1..=11 {
        let (mse_train, mse_cv) =
            linear_regression_training_with_polynomial(&data_train, &data_cv, degree);
        mse_trains.push(mse_train);
        mse_cvs.push(mse_cv);
    }
    println!("Training MSE with degrees: {mse_trains:?}");
    println!("CV MSE with degrees: {mse_cvs:?}");
    plot_mse(&mse_trains, &mse_cvs, "mse.png");

    let min_degree = mse_cvs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx + 1)
        .unwrap();

    println!("CV is minimum for degree: {min_degree}");

    let (train_mse, test_mse) =
        linear_regression_training_with_polynomial(&data_train, &data_test, min_degree);
    println!(
        "Train MSE: {}\nCV MSE: {}\nTest MSE: {}",
        train_mse,
        mse_cvs[min_degree - 1],
        test_mse
    );

    println!("Similarly we can do this in Neural Network");
}
