use csv::ReaderBuilder;
use ndarray::Array2;
use ndarray_csv::Array2Reader;
use std::path::Path;

pub fn load_rating_small() -> (Array2<f64>, Array2<f64>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data");
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_path(path.join("small_movies_Y.csv"))
        .unwrap();
    let y: Array2<f64> = rdr.deserialize_array2_dynamic().unwrap();

    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_path(path.join("small_movies_R.csv"))
        .unwrap();
    let r: Array2<f64> = rdr.deserialize_array2_dynamic().unwrap();
    (y, r)
}

pub fn load_precalc_params_small() -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data");
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_path(path.join("small_movies_X.csv"))
        .unwrap();
    let x: Array2<f64> = rdr.deserialize_array2_dynamic().unwrap();
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_path(path.join("small_movies_W.csv"))
        .unwrap();
    let w: Array2<f64> = rdr.deserialize_array2_dynamic().unwrap();
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_path(path.join("small_movies_b.csv"))
        .unwrap();
    let b: Array2<f64> = rdr.deserialize_array2_dynamic().unwrap();
    (x, w, b)
}

pub fn collaborative_filtering_cost(
    y: &Array2<f64>,
    r: &Array2<f64>,
    w: &Array2<f64>,
    x: &Array2<f64>,
    b: &Array2<f64>,
    lambda: f64,
) -> f64 {
    let mut pred = x.dot(&w.t());
    pred += b;
    let j = (pred - y) * r;
    let squared_error = 0.5 * j.mapv(|v| v.powi(2)).sum();
    let reg = (lambda / 2.0) * (x.mapv(|v| v.powi(2)).sum() + w.map(|v| v.powi(2)).sum());
    squared_error + reg
}
