use lab2_sigmoid::compute_cost_logistic;
use ndarray::array;
use plotly::ImageFormat;
use plotly::{Plot, Surface};

fn main() {
    // Training data
    let x_train = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]];
    let y_train = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    // Sample parameter space
    let n = 50; // resolution (reduce from 200 for speed)
    let ws: Vec<f64> = (0..n)
        .map(|i| -5.0 + 10.0 * (i as f64) / (n as f64))
        .collect();
    let bs: Vec<f64> = (0..n)
        .map(|i| -5.0 + 10.0 * (i as f64) / (n as f64))
        .collect();

    // Compute logistic loss surface
    let zs: Vec<Vec<f64>> = bs
        .iter()
        .map(|&b| {
            ws.iter()
                .map(|&w| compute_cost_logistic(&x_train, &y_train, &array![w], b))
                .collect()
        })
        .collect();

    // Create surface plot
    let surface = Surface::new(zs)
        .x(ws.clone())
        .y(bs.clone())
        .name("Logistic Loss Surface");

    let mut plot = Plot::new();
    plot.add_trace(surface);

    // Save to PNG (requires Orca or Kaleido installed)
    plot.write_image("logistic-loss.png", ImageFormat::PNG, 800, 600, 1.0)
        .unwrap();
}
