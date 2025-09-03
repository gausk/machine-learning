use lab1_relu::plot_xy;
use lab2_softmax::my_softmax;

fn main() {
    let z: Vec<f64> = (1..=5).map(|v| v as f64).collect();
    let y = my_softmax(&z);
    println!("Z: {:?}", z);
    println!("Softmax: {:?}", y);
    plot_xy(&z, &y, "softmax.png", "Softmax Activation");
    println!("Plot saved at softmax.png");
}
