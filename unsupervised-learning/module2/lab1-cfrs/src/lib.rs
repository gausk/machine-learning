use csv::ReaderBuilder;
use ndarray::Array;
use ndarray::{Array2, Axis};
use ndarray_csv::Array2Reader;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use serde::Deserialize;
use serde::Serialize;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MovieInfo {
    #[serde(rename = "mean rating")]
    mean_rating: f64,
    #[serde(rename = "number of ratings")]
    number_of_ratings: u64,
    pub title: String,
}

pub fn load_movie_info() -> Vec<MovieInfo> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/small_movie_list.csv");
    let mut rdr = ReaderBuilder::new().from_path(path).unwrap();
    let mut movies = Vec::new();
    for row in rdr.deserialize::<MovieInfo>() {
        let record = row.unwrap();
        movies.push(record);
    }
    movies
}

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
    let pred = x.dot(&w.t()) + b;
    let j = (pred - y) * r;
    let squared_error = 0.5 * j.mapv(|v| v.powi(2)).sum();
    let reg = (lambda / 2.0) * (x.mapv(|v| v.powi(2)).sum() + w.mapv(|v| v.powi(2)).sum());
    squared_error + reg
}

pub fn normalize_rating(y: &Array2<f64>, r: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let mean = (y * r).sum_axis(Axis(1)) / (r.sum_axis(Axis(1)) + 1e-12);
    let mean2d = mean.insert_axis(Axis(1));
    let y_mean = mean2d * r;
    let ynorm = y - &y_mean;
    (ynorm, y_mean)
}

pub fn collaborative_filtering_training(
    y_norm: &Array2<f64>,
    r: &Array2<f64>,
    lambda: f64,
    iterations: usize,
    num_features: usize,
    learning_rate: f64,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let num_movies = y_norm.shape()[0];
    let num_users = y_norm.shape()[1];
    let mut x: Array2<f64> = Array::random((num_movies, num_features), StandardNormal);
    let mut w: Array2<f64> = Array::random((num_users, num_features), StandardNormal);
    let mut b: Array2<f64> = Array::random((1, num_users), StandardNormal);

    for i in 0..iterations {
        let cost = collaborative_filtering_cost(y_norm, r, &w, &x, &b, lambda);
        let (d_x, d_w, d_b) = backward_pass(&x, &w, &b, y_norm, r, lambda);
        x -= &(learning_rate * d_x);
        w -= &(learning_rate * d_w);
        b -= &(learning_rate * d_b);

        if i % 50 == 0 {
            println!("Cost at iteration {} is {}", i, cost);
        }
    }
    (x, w, b)
}

pub fn backward_pass(
    x: &Array2<f64>,
    w: &Array2<f64>,
    b: &Array2<f64>,
    y: &Array2<f64>,
    r: &Array2<f64>,
    lambda: f64,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let pred = x.dot(&w.t()) + b;
    let e = (pred - y) * r;
    let d_x = e.dot(w) + lambda * x;
    let d_w = e.t().dot(x) + lambda * w;
    let d_b = e.sum_axis(Axis(0)).insert_axis(Axis(0));
    (d_x, d_w, d_b)
}
