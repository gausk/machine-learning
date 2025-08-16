use ndarray::array;
use plotters::prelude::*;
use std::process::Command;

fn main() {
    let x_train = array![1.0, 2.0];
    let y_train = array![300.0, 500.0];
    println!("x_train = {x_train}");
    println!("y_train = {y_train}");

    println!("x_train.shape: {}", x_train.dim());
    println!("Number of training examples: {}", x_train.len());

    let i = 0;
    let x_i = x_train[i];
    let y_i = y_train[i];
    println!("(x^({i}), y^({i})) = ({x_i}, {y_i})");

    /*
        # Plot the data points
    plt.scatter(x_train, y_train, marker='x', c='r')
    # Set the title
    plt.title("Housing Prices")
    # Set the y-axis label
    plt.ylabel('Price (in 1000s of dollars)')
    # Set the x-axis label
    plt.xlabel('Size (1000 sqft)')
    plt.show()
        */

    let root = BitMapBackend::new("model-representation.png", (500, 500)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Housing Price", ("sans-serif", 30))
        .margin(5)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(0f64..5f64, 0f64..600f64)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("Size (1000 sqft)")
        .y_desc("Price (in 1000s of dollars)")
        .draw()
        .unwrap();

    chart
        .draw_series(
            x_train
                .iter()
                .zip(y_train.iter())
                .map(|(&x, &y)| Cross::new((x, y), 10, &RED)),
        )
        .unwrap();

    Command::new("catimg")
        .arg("model-representation.png")
        .status()
        .expect("Failed to execute command");
}
