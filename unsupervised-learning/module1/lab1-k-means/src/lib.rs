use image::Rgb;
use image::{GenericImageView, ImageReader, RgbImage};
use ndarray::Array2;
use ndarray_npy::read_npy;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::path::Path;

pub fn load_data() -> Vec<Vec<f64>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/ex7_X.npy");
    let output: Array2<f64> = read_npy(path).unwrap();
    output.outer_iter().map(|row| row.to_vec()).collect()
}

pub fn find_closest_centroid(x: &[Vec<f64>], centroids: &[Vec<f64>]) -> Vec<usize> {
    let mut idxs = Vec::new();
    for row in x.iter() {
        let mut min_dist = f64::MAX;
        let mut curr_idx = 0;
        for (idx, centroid) in centroids.iter().enumerate() {
            let dist = row
                .iter()
                .zip(centroid.iter())
                .map(|(x1, x2)| (x1 - x2).powi(2))
                .sum::<f64>();
            if dist < min_dist {
                curr_idx = idx;
                min_dist = dist;
            }
        }
        idxs.push(curr_idx);
    }
    idxs
}

pub fn find_cost(x: &[Vec<f64>], centroids: &[Vec<f64>], idxs: &[usize]) -> f64 {
    let mut cost = 0.0;
    for (i, &idx) in idxs.iter().enumerate() {
        cost += centroids[idx]
            .iter()
            .zip(x[i].iter())
            .map(|(&p1, &p2)| (p1 - p2).powi(2))
            .sum::<f64>();
    }
    cost / x.len() as f64
}

pub fn compute_centroids(x: &[Vec<f64>], idxs: &[usize], centers: usize) -> Vec<Vec<f64>> {
    let mut counts = vec![0; centers];
    let mut centroids = vec![vec![0.0; x[0].len()]; centers];
    for (i, &idx) in idxs.iter().enumerate() {
        centroids[idx]
            .iter_mut()
            .zip(x[i].iter())
            .for_each(|(c1, c2)| *c1 += c2);
        counts[idx] += 1;
    }
    centroids
        .iter_mut()
        .zip(counts.iter())
        .for_each(|(centroid, count)| {
            centroid.iter_mut().for_each(|c| {
                *c /= *count as f64;
            });
        });
    centroids
}

pub fn compute_k_means(mut x: Vec<Vec<f64>>, iterations: usize, centers: usize) -> Vec<Vec<f64>> {
    x.shuffle(&mut thread_rng());
    let init_centroids: Vec<Vec<f64>> = x.clone().into_iter().take(centers).collect();
    compute_k_means_with_centroids(&x, iterations, &init_centroids)
}

pub fn compute_k_means_with_centroids(
    x: &[Vec<f64>],
    iteration: usize,
    centroids_init: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let mut centroids = centroids_init.to_vec();
    for _ in 0..iteration {
        let idxs = find_closest_centroid(x, &centroids);
        println!("cost: {}", find_cost(x, &centroids, &idxs));
        centroids = compute_centroids(x, &idxs, centroids.len());
    }
    centroids
}

pub fn load_image() -> Vec<Vec<f64>> {
    println!("Loading image bird_small.png");
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/bird_small.png");
    let img = ImageReader::open(path).unwrap().decode().unwrap();
    let dims = img.dimensions();
    println!("dimensions {:?}", dims);
    let rgb_data = img.to_rgb8();
    rgb_data
        .pixels()
        .map(|pixel| {
            vec![
                pixel[0] as f64 / 256.0,
                pixel[1] as f64 / 256.0,
                pixel[2] as f64 / 256.0,
            ]
        })
        .collect()
}

pub fn replace_pixel_with_closest_centroids(
    x: &[Vec<f64>],
    centroids: &[Vec<f64>],
) -> Vec<Vec<u8>> {
    let idxes = find_closest_centroid(x, centroids);
    let mut out: Vec<Vec<u8>> = Vec::new();
    for idx in idxes {
        out.push(
            centroids[idx]
                .iter()
                .map(|&pixel| (pixel * 256.0) as u8)
                .collect(),
        )
    }
    out
}

pub fn create_png(data: &[Vec<u8>], dimensions: (usize, usize)) {
    println!("Creating PNG ...");
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("compressed.png");
    let mut img = RgbImage::new(dimensions.0 as u32, dimensions.1 as u32);
    for width in 0..dimensions.0 {
        for height in 0..dimensions.1 {
            img.put_pixel(
                width as u32,
                height as u32,
                Rgb(data[height * dimensions.1 + width]
                    .clone()
                    .try_into()
                    .unwrap()),
            )
        }
    }
    img.save(path).unwrap();
}
