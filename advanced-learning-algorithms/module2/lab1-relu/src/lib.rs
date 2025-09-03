use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::prelude::*;

pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

pub fn plot_xy(x: &[f64], y: &[f64], path: &str, header: &str) {
    let root = BitMapBackend::new(path, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let x_min = *x.first().unwrap();
    let x_max = *x.last().unwrap();
    let y_min = y.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let mut chart = ChartBuilder::on(&root)
        .caption(header, ("sans-serif", 30))
        .margin(30)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("X")
        .y_desc("Y")
        .draw()
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            x.iter().zip(y.iter()).map(|(&x1, &y1)| (x1, y1)),
            &RED,
        ))
        .unwrap();
}
