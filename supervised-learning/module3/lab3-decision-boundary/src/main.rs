use ndarray::array;
use plotters::prelude::*;

fn plot_decision_boundary(x: &ndarray::Array2<f64>, y: &ndarray::Array1<i32>, filename: &str) {
    let root = BitMapBackend::new(filename, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Decision Boundary", ("sans-serif", 50))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-1.0..4.0, -1.0..4.0)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("Feature 1")
        .y_desc("Feature 2")
        .draw()
        .unwrap();

    // Plot data points
    for (i, point) in x.outer_iter().enumerate() {
        let color = if y[i] == 0 { BLUE } else { RED };
        chart
            .draw_series(PointSeries::of_element(
                vec![(point[0], point[1])],
                15,
                color,
                &|c, s, st| TriangleMarker::new(c, s, st),
            ))
            .unwrap();
    }

    chart
        .draw_series(LineSeries::new(
            vec![(-1.0, 4.0), (4.0, -1.0)], // two endpoints of the line
            ShapeStyle::from(&BLACK).stroke_width(2),
        ))
        .unwrap();
}

fn main() {
    let x = array![
        [0.5, 1.5],
        [1.0, 1.0],
        [1.5, 0.5],
        [3.0, 0.5],
        [2.0, 2.0],
        [1.0, 2.5]
    ];
    let y = array![0, 0, 0, 1, 1, 1];

    plot_decision_boundary(&x, &y, "decision_boundary.png");
}
