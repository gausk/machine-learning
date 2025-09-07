use itertools::Itertools;
use linfa::Dataset;
use ndarray::{Array1, Array2, ArrayView2};

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
    pub fn transform_array(&self, data: ArrayView2<f64>) -> Array2<f64> {
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

    /// Transform a linfa Dataset (records + targets)
    pub fn transform(
        &self,
        data: &Dataset<Array2<f64>, Array1<f64>>,
    ) -> Dataset<Array2<f64>, Array1<f64>> {
        let new_x = self.transform_array(data.records().view());
        Dataset::new(new_x, data.targets().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_polynomial_features() {
        let records = array![[2.0, 3.0]];
        let targets = array![1.0];
        let dataset = Dataset::new(records, targets);

        let poly_features = PolynomialFeatures::new(2, false);
        let transformed_dataset = poly_features.transform(&dataset);

        // Expected features: x1, x2, x1^2, x1*x2, x2^2
        let expected = array![[2.0, 3.0, 4.0, 6.0, 9.0]];
        assert_eq!(transformed_dataset.records(), &expected);
    }
}
