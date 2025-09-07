use itertools::Itertools;
use linfa::prelude::Fit;
use linfa::prelude::Predict;
use linfa::prelude::SingleTargetRegression;
use linfa::prelude::Transformer;
use linfa::{Dataset, DatasetBase};
use linfa_linear::LinearRegression;
use linfa_preprocessing::linear_scaling::LinearScaler;
use ndarray::{Array1, Array2};

/// PolynomialFeatures transformer for expanding features
pub struct PolynomialFeatures {
    degree: usize,
    include_bias: bool,
}

impl PolynomialFeatures {
    pub fn new(degree: usize, include_bias: bool) -> Self {
        Self {
            degree,
            include_bias,
        }
    }

    /// Transform a 2D array (features) into polynomial features
    pub fn transform_array(&self, data: &Array2<f64>) -> Array2<f64> {
        let num_samples = data.nrows();
        let num_features = data.ncols();

        if num_samples == 0 {
            return Array2::from_shape_vec((0, 0), Vec::new()).unwrap();
        }

        // Compute total number of output features
        let mut total_features = 0;
        if self.include_bias {
            total_features += 1;
        }
        for d in 1..=self.degree {
            total_features += (0..num_features).combinations_with_replacement(d).count();
        }

        let mut out = Array2::<f64>::zeros((num_samples, total_features));

        for (i, row) in data.outer_iter().enumerate() {
            let mut col = 0;

            if self.include_bias {
                out[(i, col)] = 1.0;
                col += 1;
            }

            for d in 1..=self.degree {
                for feat_indices in (0..num_features).combinations_with_replacement(d) {
                    let mut term_value = 1.0;
                    for &feat_idx in &feat_indices {
                        term_value *= row[feat_idx];
                    }
                    out[(i, col)] = term_value;
                    col += 1;
                }
            }
        }

        out
    }
}

pub fn linear_regression_training_with_polynomial(
    data_train: &DatasetBase<Array2<f64>, Array1<f64>>,
    data_cv: &DatasetBase<Array2<f64>, Array1<f64>>,
    degree: usize,
) -> (f64, f64) {
    let pl_features = PolynomialFeatures::new(degree, false);
    let train_input = pl_features.transform_array(data_train.records());

    let train_data = Dataset::new(train_input, data_train.targets.clone());
    let scaler = LinearScaler::standard().fit(&train_data).unwrap();
    let scaled_td = scaler.transform(train_data);

    let model = LinearRegression::default().fit(&scaled_td).unwrap();
    let trained_pred = model.predict(&scaled_td);
    let train_mse_error = trained_pred.mean_squared_error(&scaled_td).unwrap() / 2.0;

    let cv_input = pl_features.transform_array(data_cv.records());
    let cv_data = Dataset::new(cv_input, data_cv.targets.clone());
    let scaled_data_cv = scaler.transform(cv_data);
    let predict_cv = model.predict(&scaled_data_cv);
    let cv_mse_error = predict_cv.mean_squared_error(&scaled_data_cv).unwrap() / 2.0;
    (train_mse_error, cv_mse_error)
}

pub fn plot_mse(train_mse_error: &[f64], cv_mse_error: &[f64], path: &str) {
    use plotters::prelude::*;
    let root = BitMapBackend::new(path, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let degree = (1..=train_mse_error.len())
        .map(|x| x as f64)
        .collect::<Vec<f64>>();
    let x_min = 1.0;
    let x_max = train_mse_error.len() as f64;
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
        .x_desc("Degree")
        .y_desc("MSE")
        .y_label_offset(40)
        .label_style(("sans-serif", 15))
        .draw()
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            degree
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
            degree
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_polynomial_features() {
        let input = array![[2.0, 3.0]];
        let poly_features = PolynomialFeatures::new(2, false);
        let result = poly_features.transform_array(&input);

        // Expected features: x1, x2, x1^2, x1*x2, x2^2
        let expected = array![[2.0, 3.0, 4.0, 6.0, 9.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_polynomial_features_with_bias() {
        let input = array![[2.0, 3.0]];
        let poly_features = PolynomialFeatures::new(2, true);
        let result = poly_features.transform_array(&input);

        // Expected features: bias, x1, x2, x1^2, x1*x2, x2^2
        let expected = array![[1.0, 2.0, 3.0, 4.0, 6.0, 9.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_polynomial_features_multiple_samples() {
        let input = array![[1.0, 2.0], [3.0, 4.0]];
        let poly_features = PolynomialFeatures::new(2, false);
        let result = poly_features.transform_array(&input);

        // Expected features for each row: x1, x2, x1^2, x1*x2, x2^2
        let expected = array![
            [1.0, 2.0, 1.0, 2.0, 4.0],   // First sample: [1,2] -> [1,2,1,2,4]
            [3.0, 4.0, 9.0, 12.0, 16.0]  // Second sample: [3,4] -> [3,4,9,12,16]
        ];
        assert_eq!(result, expected);
    }
}
