use lab1_k_means::{
    compute_k_means, compute_k_means_with_centroids, create_png, load_data, load_image,
    replace_pixel_with_closest_centroids,
};

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

    let img_data = load_image();
    println!(
        "Image Data shape: [{}, {}]",
        img_data.len(),
        img_data[0].len()
    );
    println!("First Data: {:?}", img_data[0]);
    let centroids = compute_k_means(img_data.clone(), 10, 16);
    println!("Centroids: {:?}", centroids);
    let compressed_data = replace_pixel_with_closest_centroids(&img_data, &centroids);
    create_png(&compressed_data, (128, 128));
    println!("Image compression complete!");
}
