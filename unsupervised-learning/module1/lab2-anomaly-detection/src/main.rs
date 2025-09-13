use lab2_anomaly_detection::{
    calculate_mean_and_variance, load_data, multivariate_gaussian, select_threshold,
};

fn main() {
    println!("Welcome to Lab2 on Anomaly Detection!");
    println!("Let's try on simple dataset with 2 features.");
    let (x_train, x_val, y_val) = load_data("part1");
    println!("x_train shape: {:?}", x_train.shape());
    println!("x_val shape: {:?}", x_val.shape());
    println!("y_val shape: {:?}", y_val.shape());
    let (mu, var) = calculate_mean_and_variance(&x_train);
    println!("mu: {:?}", mu.to_vec());
    println!("var: {:?}", var.to_vec());
    let probabilities = multivariate_gaussian(&x_val, &mu, &var);
    let (epsilon, f1) = select_threshold(&probabilities, &y_val);
    println!("epsilon: {:?}", epsilon);
    println!("f1: {:?}", f1);

    println!("Let's try on dataset with multiple features.");
    let (x_train, x_val, y_val) = load_data("part2");
    println!("x_train shape: {:?}", x_train.shape());
    println!("x_val shape: {:?}", x_val.shape());
    println!("y_val shape: {:?}", y_val.shape());
    let (mu, var) = calculate_mean_and_variance(&x_train);
    println!("mu: {:?}", mu.to_vec());
    println!("var: {:?}", var.to_vec());
    let probabilities = multivariate_gaussian(&x_val, &mu, &var);
    let (epsilon, f1) = select_threshold(&probabilities, &y_val);
    println!("epsilon: {:?}", epsilon);
    println!("f1: {:?}", f1);
}
