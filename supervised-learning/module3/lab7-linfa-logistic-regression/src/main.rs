use linfa::prelude::*;
use linfa_logistic::LogisticRegression;
use ndarray::array;

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

    let dataset = Dataset::new(x, y.clone());
    let model = LogisticRegression::default().fit(&dataset).unwrap();
    let pred = model.predict(&dataset);
    println!("Actual: {} Predictions: {}", y, pred);
    println!("Model parameters: {}", model.params());
    println!("Model intercept: {}", model.intercept());

    let accuracy = pred.confusion_matrix(&y).unwrap().accuracy();
    println!("Accuracy: {}", accuracy);
}
