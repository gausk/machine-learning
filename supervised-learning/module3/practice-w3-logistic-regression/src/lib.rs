use csv::ReaderBuilder;
use lab2_sigmoid::sigmoid;
use ndarray::{Array1, Array2, s};

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

pub fn predict_logistic(x: &Array2<f64>, w: &Array1<f64>, b: f64) -> Array1<f64> {
    let z = x.dot(w) + b;
    let p = sigmoid(&z);
    p.mapv(|v| if v >= 0.5 { 1.0 } else { 0.0 })
}

pub fn map_feature(x1: &Array1<f64>, x2: &Array1<f64>) -> Array2<f64> {
    let degree = 6;
    let m = x1.len();
    let mut features: Vec<Array1<f64>> = Vec::new();

    for i in 1..=degree {
        for j in 0..=i {
            // (x1^(i-j)) * (x2^j)
            let col: Array1<f64> =
                x1.mapv(|v| v.powi(i - j)) * x2.mapv(|v| v.powi(j));
            features.push(col);
        }
    }

    // Stack horizontally into Array2
    let mut out = Array2::<f64>::zeros((m, features.len()));
    for (j, col) in features.into_iter().enumerate() {
        out.slice_mut(s![.., j]).assign(&col);
    }

    out
}
