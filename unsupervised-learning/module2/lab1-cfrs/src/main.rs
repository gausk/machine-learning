use lab1_cfrs::{collaborative_filtering_cost, load_precalc_params_small, load_rating_small};
use ndarray::s;

fn main() {
    println!("Welcome to Lab on Collaborative Filtering Recommender Systems");
    let (y, r) = load_rating_small();
    println!("Y shape: {:?}", y.shape());
    println!("R shape: {:?}", r.shape());
    let (x, w, b) = load_precalc_params_small();
    println!("X shape: {:?}", x.shape());
    println!("W shape: {:?}", w.shape());
    println!("B shape: {:?}", b.shape());
    let num_features = w.shape()[1];
    let num_movies = y.shape()[0];
    let num_users = y.shape()[1];
    println!("num_movies: {}", num_movies);
    println!("num_users: {}", num_users);
    println!("num_features: {}", num_features);

    let y_row = y.row(0);
    let r_row = r.row(0);
    let selected: Vec<f64> = y_row
        .iter()
        .zip(r_row.iter())
        .filter(|&(_, &z1)| z1 != 0.0)
        .map(|(&val, _)| val)
        .collect();
    let tsmean = selected.iter().sum::<f64>() / selected.len() as f64;
    println!("tsmean: {}", tsmean);

    let num_users_r = 4;
    let num_movies_r = 5;
    let num_features_r = 3;
    let cost_without_reg = collaborative_filtering_cost(
        &y.slice(s![0..num_movies_r, 0..num_users_r]).to_owned(),
        &r.slice(s![0..num_movies_r, 0..num_users_r]).to_owned(),
        &w.slice(s![0..num_users_r, 0..num_features_r]).to_owned(),
        &x.slice(s![0..num_movies_r, 0..num_features_r]).to_owned(),
        &b.slice(s![0..1, 0..num_users_r]).to_owned(),
        0.0,
    );
    println!("cost_with_reg: {}", cost_without_reg);
    let cost_with_reg = collaborative_filtering_cost(
        &y.slice(s![0..num_movies_r, 0..num_users_r]).to_owned(),
        &r.slice(s![0..num_movies_r, 0..num_users_r]).to_owned(),
        &w.slice(s![0..num_users_r, 0..num_features_r]).to_owned(),
        &x.slice(s![0..num_movies_r, 0..num_features_r]).to_owned(),
        &b.slice(s![0..1, 0..num_users_r]).to_owned(),
        1.5,
    );
    println!("cost_with_reg: {}", cost_with_reg);
}
