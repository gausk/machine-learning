use itertools::Itertools;
use ndarray::Array2;

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
