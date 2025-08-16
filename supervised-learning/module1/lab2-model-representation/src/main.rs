use ndarray::array;
use plotters::prelude::*;

fn compute_model_output(x: &ndarray::Array1<f64>, w: f64, b: f64) -> ndarray::Array1<f64> {
    let m = x.len();
    let mut f_wb = ndarray::Array1::zeros(m);
    for i in 0..m {
        f_wb[i] = w * x[i] + b;
    }
    f_wb
}

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

    let root = BitMapBackend::new("model-representation.png", (500, 500)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Housing Price", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(0.8f64..2.2f64, 180f64..520f64)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("Size (1000 sqft)")
        .y_desc("Price (in 1000s of dollars)")
        .label_style(("sans-serif", 15))
        .y_label_offset(30)
        .draw()
        .unwrap();

    chart
        .draw_series(
            x_train
                .iter()
                .zip(y_train.iter())
                .map(|(&x, &y)| Cross::new((x, y), 10, RED)),
        )
        .unwrap()
        .label("Actual Values")
        .legend(|(x, y)| Cross::new((x + 5, y), 5, RED));

    let w = 100;
    let b = 100;
    println!("w: {w}");
    println!("b: {b}");

    let tmp_f_wb = compute_model_output(&x_train, w as f64, b as f64);
    println!("tmp_f_wb = {tmp_f_wb}");

    chart
        .draw_series(LineSeries::new(
            x_train.iter().zip(tmp_f_wb.iter()).map(|(&x, &y)| (x, y)),
            BLUE,
        ))
        .unwrap()
        .label("Our Prediction")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], BLUE));

    chart
        .configure_series_labels()
        .background_style(WHITE)
        .border_style(BLACK)
        .position(SeriesLabelPosition::UpperLeft)
        .draw()
        .unwrap();

    root.present().unwrap();
    println!("Plot saved to model-representation.png");

    let w = 200.0;
    let b = 100.0;
    let x_i = 1.2;
    let cost_1200sqft = w * x_i + b;

    println!("${:.2} thousand dollars", cost_1200sqft);
}
