use lab1_model_evaluation::{linear_regression_training_with_polynomial, plot_mse};
use lab2_bias_and_variance::{linear_regression_with_regularization, plot_mse_with_regularization};
use lab3_fs_and_lr::load_data;
use linfa::Dataset;
use linfa::dataset::Records;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::path::Path;

fn main() {
    println!("Welcome to lab2 on Bias and Variance");
    println!("MSE for Linear Regression with one feature");
    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/c2w2_lab2_data1.csv");
    let (x, y) = load_data(data_path.to_str().unwrap(), 1);

    let data = Dataset::new(x, y);
    println!("Input data shape: {:?}", data.nsamples());
    let mut rng = StdRng::seed_from_u64(42);
    let (data_train, data_test_and_validate) = data.shuffle(&mut rng).split_with_ratio(0.6);
    let (data_test, data_cv) = data_test_and_validate.split_with_ratio(0.5);
    println!("Train data shape: {:?}", data_train.nsamples());
    println!("Test data shape: {:?}", data_test.nsamples());
    println!("Cross validation data shape: {:?}", data_cv.nsamples());

    let mut mse_trains = Vec::new();
    let mut mse_cvs = Vec::new();
    for degree in 1..=10 {
        let (mse_train, mse_cv) =
            linear_regression_training_with_polynomial(&data_train, &data_cv, degree);
        mse_trains.push(mse_train);
        mse_cvs.push(mse_cv);
    }
    println!("Training MSE with degrees: {mse_trains:?}");
    println!("CV MSE with degrees: {mse_cvs:?}");
    plot_mse(&mse_trains, &mse_cvs, "mse_with_one_feature.png");

    let (min_degree, min_mse) = mse_cvs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, &x)| (idx + 1, x))
        .unwrap();

    println!("MSE is minimum for CV at degree: {min_degree} with value: {min_mse}");

    let (min_degree, min_mse) = mse_trains
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, &x)| (idx + 1, x))
        .unwrap();
    println!("MSE is minimum for training data at degree: {min_degree} with value: {min_mse}");
    println!(
        "\nIf baseline error is around {min_mse}, then great or if it is lower than we have to add more features."
    );
    println!("MSE for Linear Regression with two features");

    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/c2w2_lab2_data2.csv");
    let (x, y) = load_data(data_path.to_str().unwrap(), 2);

    let data = Dataset::new(x, y);
    println!("Input data shape: {:?}", data.nsamples());
    let mut rng = StdRng::seed_from_u64(42);
    let (data_train, data_test_and_validate) = data.shuffle(&mut rng).split_with_ratio(0.6);
    let (data_test, data_cv) = data_test_and_validate.split_with_ratio(0.5);
    println!("Train data shape: {:?}", data_train.nsamples());
    println!("Test data shape: {:?}", data_test.nsamples());
    println!("Cross validation data shape: {:?}", data_cv.nsamples());

    let mut mse_trains = Vec::new();
    let mut mse_cvs = Vec::new();
    for degree in 1..=6 {
        let (mse_train, mse_cv) =
            linear_regression_training_with_polynomial(&data_train, &data_cv, degree);
        mse_trains.push(mse_train);
        mse_cvs.push(mse_cv);
    }
    println!("Training MSE with degrees: {mse_trains:?}");
    println!("CV MSE with degrees: {mse_cvs:?}");
    plot_mse(&mse_trains, &mse_cvs, "mse_with_two_feature.png");

    let (min_degree, min_mse) = mse_cvs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, &x)| (idx + 1, x))
        .unwrap();

    println!("MSE is minimum for CV at degree: {min_degree} with value: {min_mse}");

    let (min_degree, min_mse) = mse_trains
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, &x)| (idx + 1, x))
        .unwrap();
    println!("MSE is minimum for training data at degree: {min_degree} with value: {min_mse}");

    println!("Now training error is lower than baseline");
    println!("\nWe have to add regularization to avoid over fitting");

    let regularizations = vec![
        10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.1,
    ];
    let mut mse_trains = Vec::new();
    let mut mse_cvs = Vec::new();
    for lambda in &regularizations {
        let (mse_train, mse_cv) =
            linear_regression_with_regularization(&data_train, &data_cv, 4, *lambda);
        mse_trains.push(mse_train);
        mse_cvs.push(mse_cv);
    }
    println!("Training MSE with regularization: {mse_trains:?}");
    println!("CV MSE with regularization: {mse_cvs:?}");
    let (min_lambda, min_mse) = mse_cvs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, &x)| (regularizations[idx], x))
        .unwrap();

    println!("MSE is minimum for CV at lambda: {min_lambda} with value: {min_mse}");

    let (min_lambda, min_mse) = mse_trains
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, &x)| (regularizations[idx], x))
        .unwrap();
    println!("MSE is minimum for training data at lambda: {min_lambda} with value: {min_mse}");
    plot_mse_with_regularization(
        &regularizations,
        &mse_trains,
        &mse_cvs,
        "mse_with_regularization.png",
    );

    println!("Trying smaller regularization value");
    let regularizations = vec![
        0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    ];

    let mut mse_trains = Vec::new();
    let mut mse_cvs = Vec::new();
    for lambda in &regularizations {
        let (mse_train, mse_cv) =
            linear_regression_with_regularization(&data_train, &data_cv, 4, *lambda);
        mse_trains.push(mse_train);
        mse_cvs.push(mse_cv);
    }
    println!("\nTraining MSE with regularization: {mse_trains:?}");
    println!("CV MSE with regularization: {mse_cvs:?}");
    let (min_lambda, min_mse) = mse_cvs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, &x)| (regularizations[idx], x))
        .unwrap();

    println!("MSE is minimum for CV at lambda: {min_lambda} with value: {min_mse}");

    let (min_lambda, min_mse) = mse_trains
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, &x)| (regularizations[idx], x))
        .unwrap();
    println!("MSE is minimum for training data at lambda: {min_lambda} with value: {min_mse}");
    plot_mse_with_regularization(
        &regularizations,
        &mse_trains,
        &mse_cvs,
        "mse_with_small_regularization.png",
    );

    println!("\nTry adding random feature");
    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/c2w2_lab2_data3.csv");
    let (x, y) = load_data(data_path.to_str().unwrap(), 3);

    let data = Dataset::new(x, y);
    println!(
        "Input data shape: {}, features: {}",
        data.nsamples(),
        data.nfeatures()
    );
    let mut rng = StdRng::seed_from_u64(42);
    let (data_train, data_test_and_validate) = data.shuffle(&mut rng).split_with_ratio(0.6);
    let (data_test, data_cv) = data_test_and_validate.split_with_ratio(0.5);
    println!("Train data shape: {:?}", data_train.nsamples());
    println!("Test data shape: {:?}", data_test.nsamples());
    println!("Cross validation data shape: {:?}", data_cv.nsamples());

    let mut mse_trains = Vec::new();
    let mut mse_cvs = Vec::new();
    for degree in 1..=4 {
        let (mse_train, mse_cv) =
            linear_regression_training_with_polynomial(&data_train, &data_cv, degree);
        mse_trains.push(mse_train);
        mse_cvs.push(mse_cv);
    }
    println!("Training MSE with degrees: {mse_trains:?}");
    println!("CV MSE with degrees: {mse_cvs:?}");
    plot_mse(&mse_trains, &mse_cvs, "mse_with_three_feature.png");

    let (min_degree, min_mse) = mse_cvs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, &x)| (idx + 1, x))
        .unwrap();

    println!("MSE is minimum for CV at degree: {min_degree} with value: {min_mse}");

    let (min_degree, min_mse) = mse_trains
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, &x)| (idx + 1, x))
        .unwrap();
    println!("MSE is minimum for training data at degree: {min_degree} with value: {min_mse}");

    println!("\nTry for larger dataset");
    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/c2w2_lab2_data4.csv");
    let (x, y) = load_data(data_path.to_str().unwrap(), 2);

    let data = Dataset::new(x, y);
    println!(
        "Input data shape: {}, features: {}",
        data.nsamples(),
        data.nfeatures()
    );
    let mut rng = StdRng::seed_from_u64(42);
    let (data_train, data_test_and_validate) = data.shuffle(&mut rng).split_with_ratio(0.6);
    let (data_test, data_cv) = data_test_and_validate.split_with_ratio(0.5);
    println!("Train data shape: {:?}", data_train.nsamples());
    println!("Test data shape: {:?}", data_test.nsamples());
    println!("Cross validation data shape: {:?}", data_cv.nsamples());

    let mut mse_trains = Vec::new();
    let mut mse_cvs = Vec::new();
    for degree in 1..=4 {
        let (mse_train, mse_cv) =
            linear_regression_training_with_polynomial(&data_train, &data_cv, degree);
        mse_trains.push(mse_train);
        mse_cvs.push(mse_cv);
    }
    println!("Training MSE with degrees: {mse_trains:?}");
    println!("CV MSE with degrees: {mse_cvs:?}");
    plot_mse(&mse_trains, &mse_cvs, "mse_with_large_dataset.png");

    let (min_degree, min_mse) = mse_cvs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, &x)| (idx + 1, x))
        .unwrap();

    println!("MSE is minimum for CV at degree: {min_degree} with value: {min_mse}");

    let (min_degree, min_mse) = mse_trains
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, &x)| (idx + 1, x))
        .unwrap();
    println!("MSE is minimum for training data at degree: {min_degree} with value: {min_mse}");
}
