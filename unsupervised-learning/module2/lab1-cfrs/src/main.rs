use lab1_cfrs::{
    collaborative_filtering_cost, collaborative_filtering_training, load_movie_info,
    load_precalc_params_small, load_rating_small, normalize_rating,
};
use ndarray::{Array1, Axis, s};

fn main() {
    println!("Welcome to Lab on Collaborative Filtering Recommender Systems");
    let (mut y, mut r) = load_rating_small();
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
    let movies = load_movie_info();
    println!("First Movie: {:?}", movies[0]);

    let mut my_ratings = Array1::<f64>::zeros(num_movies);
    let mut my_r = Array1::<f64>::zeros(num_movies);
    my_ratings[2700] = 5.0; // Toy Story 3 (2010)
    my_ratings[2609] = 2.0; // Persuasion (2007)
    my_ratings[929] = 5.0; // Lord of the Rings: The Return of the King, The
    my_ratings[246] = 5.0; // Shrek (2001)
    my_ratings[2716] = 3.0; // Inception
    my_ratings[1150] = 5.0; // Incredibles, The (2004)
    my_ratings[382] = 2.0; // Amelie (Fabuleux destin d'Amélie Poulain, Le)
    my_ratings[366] = 5.0; // Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)
    my_ratings[622] = 5.0; // Harry Potter and the Chamber of Secrets (2002)
    my_ratings[988] = 3.0; // Eternal Sunshine of the Spotless Mind (2004)
    my_ratings[2925] = 1.0; // Louis Theroux: Law & Disorder (2008)
    my_ratings[2937] = 1.0; // # Nothing to Declare (Rien à déclarer)
    my_ratings[793] = 5.0; // Pirates of the Caribbean: The Curse of the Black Pearl (2003)

    let mut my_rated = vec![];
    for i in 0..num_movies {
        if my_ratings[i] > 0.0 {
            my_rated.push(i);
            my_r[i] = 1.0;
            println!("My rating {} for movie {}", my_ratings[i], movies[i].title);
        }
    }
    y.push(Axis(1), (&my_ratings).into()).unwrap();
    r.push(Axis(1), (&my_r).into()).unwrap();
    println!("y new shape: {:?}", y.shape());
    println!("r new shae: {:?}", r.shape());
    let (y_norm, y_mean) = normalize_rating(&y, &r);
    let (x_o, w_o, b_o) = collaborative_filtering_training(&y_norm, &r, 1.0, 1000, 100, 0.001);
    println!("x_o shape: {:?}", x_o.shape());
    println!("w_o shape: {:?}", w_o.shape());
    println!("b_o shape: {:?}", b_o.shape());

    let predict = x_o.dot(&w_o.t()) + b_o + y_mean;
    println!("predict shape {:?}", predict.shape());

    for idx in my_rated {
        println!(
            "Predicted rating: {}, actual rating: {} for movie {}",
            y[[idx, num_users]],
            predict[[idx, num_users]],
            movies[idx].title
        );
    }
}
