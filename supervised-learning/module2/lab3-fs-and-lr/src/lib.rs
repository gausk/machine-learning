use csv::ReaderBuilder;
use ndarray::Axis;
use ndarray::{Array1, Array2};

pub fn load_data(path: &str, x_len: usize) -> (Array2<f64>, Array1<f64>) {
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .unwrap();
    let mut x = Vec::new();
    let mut y = Vec::new();

    for result in rdr.records() {
        let record = result.unwrap();
        let features: Vec<f64> = record
            .iter()
            .take(x_len)
            .map(|s| s.parse().unwrap())
            .collect();
        let target: f64 = record[x_len].parse().unwrap();
        x.push(features);
        y.push(target);
    }

    (
        Array2::from_shape_vec((x.len(), x_len), x.concat()).unwrap(),
        Array1::from_vec(y),
    )
}

pub fn zscore_normalize_features(x: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mu = x.mean_axis(Axis(0)).unwrap();
    let sigma = x.std_axis(Axis(0), 0.0);
    let x_norm = (x - &mu) / &sigma;

    (x_norm, mu, sigma)
}
