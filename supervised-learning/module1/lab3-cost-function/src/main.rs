use ndarray::{Array1, array};
use plotters::prelude::*;

/// Computes the cost for linear regression
fn compute_cost(x_train: &Array1<f64>, y_train: Array1<f64>, w: f64, b: f64) -> f64 {
    let m = x_train.len();
    let cost_sum: f64 = x_train
        .iter()
        .zip(y_train.iter())
        .map(|(&x, &y)| (w * x + b - y).powi(2))
        .sum();
    cost_sum / (2 * m) as f64
}

fn plot_house(
    x_train: &Array1<f64>,
    y_train: &Array1<f64>,
    w: f64,
    b: f64,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_min = x_train.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x_train.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = y_train.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y_train.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Housing Prices", ("sans-serif", 30))
        .margin(30)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Size (1000 sqft)")
        .y_desc("Price (in 1000s of dollars)")
        .y_label_offset(30)
        .draw()?;

    // Draw actual values
    chart
        .draw_series(
            x_train
                .iter()
                .zip(y_train.iter())
                .map(|(&x, &y)| Circle::new((x, y), 5, RED.filled())),
        )?
        .label("Actual Value")
        .legend(|(x, y)| Circle::new((x + 5, y), 5, RED.filled()));

    // Draw prediction line
    chart
        .draw_series(LineSeries::new(
            x_train.iter().map(|&x| (x, w * x + b)),
            BLUE,
        ))
        .unwrap()
        .label("Our Prediction")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], BLUE));

    // Draw dotted lines and diff values
    chart
        .draw_series(x_train.iter().zip(y_train.iter()).map(|(&x, &y)| {
            let y_pred = w * x + b;
            PathElement::new(
                vec![(x, y), (x, y_pred)],
                ShapeStyle {
                    filled: false,
                    color: GREEN.into(),
                    stroke_width: 2,
                },
            )
        }))?
        .label("Error (Actual-Predicted)")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x + 5, y - 5), (x + 5, y + 5)],
                ShapeStyle {
                    filled: false,
                    color: GREEN.into(),
                    stroke_width: 2,
                },
            )
        });

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .position(SeriesLabelPosition::UpperLeft)
        .draw()?;

    Ok(())
}

fn plot_cost_function(
    x_train: &Array1<f64>,
    y_train: &Array1<f64>,
    b: f64,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let w_start = -100.0;
    let w_end = 400.0;
    let step = 5.0;
    let mut points = Vec::new();
    let mut min_cost = f64::INFINITY;
    let mut min_w = w_start;
    let mut w = w_start;
    while w <= w_end {
        let cost = compute_cost(x_train, y_train.clone(), w, b);
        if cost < min_cost {
            min_cost = cost;
            min_w = w;
        }
        points.push((w, cost));
        w += step;
    }

    let y_min = points
        .iter()
        .map(|&(_, cost)| cost)
        .fold(f64::INFINITY, f64::min);
    let y_max = points
        .iter()
        .map(|&(_, cost)| cost)
        .fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Cost Function", ("sans-serif", 30))
        .margin(30)
        .x_label_area_size(40)
        .y_label_area_size(70)
        .build_cartesian_2d(w_start..w_end, y_min - 10000.0..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Weight (w)")
        .y_desc("Cost")
        .draw()?;

    // Draw the minimum cost vertical line with increased width
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(min_w, y_min), (min_w, y_min - 10000.0)],
        ShapeStyle {
            filled: false,
            color: BLACK.into(),
            stroke_width: 2,
        },
    )))?;

    // Show the min_w value on the x-axis
    chart.draw_series(std::iter::once(Text::new(
        format!("min_w = {:.2}", min_w),
        (min_w + 10.0, y_min - 3000.0),
        ("sans-serif", 10).into_font().color(&BLACK),
    )))?;

    // Draw the cost function as a line
    chart.draw_series(LineSeries::new(points, &RED))?;

    Ok(())
}

fn main() {
    let x_train = array![1.0, 1.7, 2.0, 2.5, 3.0, 3.2];
    let y_train = array![250.0, 300.0, 480.0, 430.0, 630.0, 730.0];
    println!("x_train = {x_train}");
    println!("y_train = {y_train}");

    println!("x_train.shape: {}", x_train.dim());
    println!("Number of training examples: {}", x_train.len());

    let w = 100f64;
    let b = 200f64;

    // Plot and save to file
    if let Err(e) = plot_house(&x_train, &y_train, w, b, "plot-data.png") {
        eprintln!("Plot error: {e}");
    }

    if let Err(e) = plot_cost_function(&x_train, &y_train, b, "plot-cost-function.png") {
        eprintln!("Cost function plot error: {e}");
    }
}
