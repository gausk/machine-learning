use lab1_k_means::{compute_k_means, compute_k_means_with_centroids, load_data};

fn main() {
    println!("Welcome to Lab1 on K Means!");
    let data = load_data();
    println!("Data shape: [{}, {}]", data.len(), data[0].len());
    let fixed_centriods = vec![vec![3.0, 3.0], vec![6.0, 2.0], vec![8.0, 5.0]];
    println!("Trying K-Mean with fixed centers");
    let centroids = compute_k_means_with_centroids(&data, 10, &fixed_centriods);
    println!("Centroids: {:?}", centroids);
    println!("Trying K-Mean with random centroids");
    let centroids = compute_k_means(data, 10, 3);
    println!("Centroids: {:?}", centroids);
}
