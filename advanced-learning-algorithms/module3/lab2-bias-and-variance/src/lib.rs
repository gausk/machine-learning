use lab1_model_evaluation::PolynomialFeatures;
use linfa::prelude::Fit;
use linfa::prelude::Predict;
use linfa::prelude::SingleTargetRegression;
use linfa::prelude::Transformer;
use linfa::{Dataset, DatasetBase};
use linfa_elasticnet::ElasticNet;
use linfa_preprocessing::linear_scaling::LinearScaler;
use ndarray::{Array1, Array2};

pub fn linear_regression_with_regularization(
    data_train: &DatasetBase<Array2<f64>, Array1<f64>>,
    data_cv: &DatasetBase<Array2<f64>, Array1<f64>>,
    degree: usize,
    lambda: f64,
) -> (f64, f64) {
    let pl_features = PolynomialFeatures::new(degree, false);
    let train_input = pl_features.transform_array(data_train.records());

    let train_data = Dataset::new(train_input, data_train.targets.clone());
    let scaler = LinearScaler::standard().fit(&train_data).unwrap();
    let scaled_td = scaler.transform(train_data);

    let model = ElasticNet::params()
        .penalty(lambda)
        .l1_ratio(0.0)
        .fit(&scaled_td)
        .unwrap();
    let trained_pred = model.predict(&scaled_td);
    let train_mse_error = trained_pred.mean_squared_error(&scaled_td).unwrap() / 2.0;

    let cv_input = pl_features.transform_array(data_cv.records());
    let cv_data = Dataset::new(cv_input, data_cv.targets.clone());
    let scaled_data_cv = scaler.transform(cv_data);
    let predict_cv = model.predict(&scaled_data_cv);
    let cv_mse_error = predict_cv.mean_squared_error(&scaled_data_cv).unwrap() / 2.0;
    (train_mse_error, cv_mse_error)
}

pub fn plot_mse_with_regularization(
    regularizations: &[f64],
    train_mse_error: &[f64],
    cv_mse_error: &[f64],
    path: &str,
) {
    use plotters::prelude::*;
    let root = BitMapBackend::new(path, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let (x_min, x_max) = regularizations
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), x| {
            (min.min(*x), max.max(*x))
        });
    let (y_min, y_max) = train_mse_error.iter().zip(cv_mse_error.iter()).fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(min, max), (&a, &p)| (min.min(a.min(p)), max.max(a.max(p))),
    );

    let mut chart = ChartBuilder::on(&root)
        .caption("Mean Squared Error", ("sans-serif", 30))
        .margin(5)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("Regularization")
        .y_desc("MSE")
        .y_label_offset(40)
        .label_style(("sans-serif", 15))
        .draw()
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            regularizations
                .iter()
                .zip(train_mse_error.iter())
                .map(|(x, y)| (*x, *y)),
            ShapeStyle {
                color: RED.mix(1.0),
                filled: false,
                stroke_width: 3,
            },
        ))
        .unwrap()
        .label("Training MSE Error")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 8, y)], RED));

    chart
        .draw_series(LineSeries::new(
            regularizations
                .iter()
                .zip(cv_mse_error.iter())
                .map(|(x, y)| (*x, *y)),
            ShapeStyle {
                color: BLUE.mix(1.0),
                filled: false,
                stroke_width: 3,
            },
        ))
        .unwrap()
        .label("CV MSE Error")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 8, y)], BLUE));

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .draw()
        .unwrap();
}
