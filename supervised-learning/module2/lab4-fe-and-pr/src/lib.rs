use ndarray::Array1;

pub fn plot_xy_actual_predicted(
    x: &Array1<f64>,
    y_actual: &Array1<f64>,
    y_predicted: &Array1<f64>,
    path: &str,
    header: &str,
) {
    use plotters::prelude::*;
    let root = BitMapBackend::new(path, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let (y_min, y_max) = y_actual.iter().zip(y_predicted.iter()).fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(min, max), (&a, &p)| (min.min(a.min(p)), max.max(a.max(p))),
    );

    let mut chart = ChartBuilder::on(&root)
        .caption(header, ("sans-serif", 30))
        .margin(5)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("x")
        .y_desc("y")
        .y_label_offset(40)
        .label_style(("sans-serif", 15))
        .draw()
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            x.iter().zip(y_actual.iter()).map(|(x, y)| (*x, *y)),
            ShapeStyle {
                color: RED.mix(1.0),
                filled: false,
                stroke_width: 3,
            },
        ))
        .unwrap()
        .label("Actual")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 8, y)], RED));

    chart
        .draw_series(LineSeries::new(
            x.iter().zip(y_predicted.iter()).map(|(x, y)| (*x, *y)),
            ShapeStyle {
                color: BLUE.mix(1.0),
                filled: false,
                stroke_width: 3,
            },
        ))
        .unwrap()
        .label("Predicted")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 8, y)], BLUE));

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .draw()
        .unwrap();
}
