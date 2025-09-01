use lab1_neuron_and_layers::SampleData;
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

/// Creates a coffee roasting dataset.
/// - roasting duration: 12-15 minutes is best
/// - temperature range: 175-260C is best
pub fn load_coffee_data(n: usize) -> Vec<SampleData> {
    let mut rng = StdRng::seed_from_u64(2);
    let mut dataset = Vec::with_capacity(n);

    for _ in 0..n {
        let mut t = rng.r#gen::<f32>(); // temperature raw [0,1]
        let mut d = rng.r#gen::<f32>(); // duration raw [0,1]

        d = d * 4.0 + 11.5; // roasting duration: 12-15
        t = t * (285.0 - 150.0) + 150.0; // temperature: 150-285

        // classification condition
        let y_line = -3.0 / (260.0 - 175.0) * t + 21.0;
        let label = if t > 175.0 && t < 260.0 && d > 12.0 && d < 15.0 && d <= y_line {
            1.0
        } else {
            0.0
        };

        dataset.push(SampleData {
            input: vec![t, d],
            target: vec![label],
        });
    }

    dataset
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationStats {
    pub input_means: Vec<f32>,
    pub input_stds: Vec<f32>,
}

impl NormalizationStats {
    pub fn new(input_means: Vec<f32>, input_stds: Vec<f32>) -> Self {
        Self {
            input_means,
            input_stds,
        }
    }

    pub fn load_from_file(path: &str) -> Self {
        let stats_json = std::fs::read_to_string(path).expect("Failed to read normalization stats");
        serde_json::from_str(&stats_json).expect("Failed to parse normalization stats")
    }

    pub fn save_to_file(&self, path: &str) {
        let stats_json =
            serde_json::to_string(&self).expect("Failed to serialize normalization stats");
        std::fs::write(path, stats_json).expect("Failed to save normalization stats");
    }
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
