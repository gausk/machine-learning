use lab1_neuron_and_layers::SampleData;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationStats {
    pub input_means: Vec<f32>,
    pub input_stds: Vec<f32>,
}

// Normalize each input feature separately and return stats
pub fn normalize_inputs_with_stats(data: &mut [SampleData]) -> NormalizationStats {
    if data.is_empty() {
        return NormalizationStats {
            input_means: vec![],
            input_stds: vec![],
        };
    }

    let num_features = data[0].input.len();
    let mut input_means = vec![0.0; num_features];
    let mut input_stds = vec![0.0; num_features];

    for feature_idx in 0..num_features {
        let feature_values: Vec<f32> = data.iter().map(|s| s.input[feature_idx]).collect();

        let mean = feature_values.iter().sum::<f32>() / feature_values.len() as f32;
        let std = (feature_values
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>()
            / feature_values.len() as f32)
            .sqrt()
            .max(1e-8);

        input_means[feature_idx] = mean;
        input_stds[feature_idx] = std;

        for sample in data.iter_mut() {
            sample.input[feature_idx] = (sample.input[feature_idx] - mean) / std;
        }

        println!(
            "Feature {} normalized: mean={:.2}, std={:.2}",
            feature_idx, mean, std
        );
    }

    NormalizationStats {
        input_means,
        input_stds,
    }
}

// Apply saved normalization stats to new data
pub fn apply_input_normalization(data: &mut [SampleData], stats: &NormalizationStats) {
    for sample in data.iter_mut() {
        for (i, value) in sample.input.iter_mut().enumerate() {
            if i < stats.input_means.len() {
                *value = (*value - stats.input_means[i]) / stats.input_stds[i];
            }
        }
    }
}

// Normalize a single input vector (for inference)
pub fn normalize_single_input(input: &mut [f32], stats: &NormalizationStats) {
    for (i, value) in input.iter_mut().enumerate() {
        if i < stats.input_means.len() {
            *value = (*value - stats.input_means[i]) / stats.input_stds[i];
        }
    }
}

// Normalize each target feature separately (for regression only)
pub fn normalize_regression_targets(data: &mut [SampleData]) {
    if data.is_empty() {
        return;
    }

    let num_targets = data[0].target.len();

    for target_idx in 0..num_targets {
        let target_values: Vec<f32> = data.iter().map(|s| s.target[target_idx]).collect();
        let mean = target_values.iter().sum::<f32>() / target_values.len() as f32;
        let std = (target_values
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>()
            / target_values.len() as f32)
            .sqrt()
            .max(1e-8);

        for sample in data.iter_mut() {
            sample.target[target_idx] = (sample.target[target_idx] - mean) / std;
        }

        println!(
            "Target {} normalized: mean={:.2}, std={:.2}",
            target_idx, mean, std
        );
    }
}
