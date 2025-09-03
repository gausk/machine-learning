use lab1_relu::{plot_xy, relu};

fn main() {
    println!("Welcome to Lab1 on ReLU activation");

    let x: Vec<f64> = (-20..=20).map(|v| v as f64).collect();
    let y: Vec<f64> = x.iter().map(|&v| relu(v)).collect();

    plot_xy(&x, &y, "relu.png", "ReLU Activation");
    println!("Plot saved to relu.png");
}
