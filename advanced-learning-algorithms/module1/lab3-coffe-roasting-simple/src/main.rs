use lab2_coffe_roasting::{
    NormalizationStats, apply_input_normalization, load_coffee_data, normalize_single_input,
};
use lab2_sigmoid::sigmoid_function;

fn my_dense(a_in: &[f32], w: &[Vec<f32>], b: &[f32]) -> Vec<f32> {
    let units = w[0].len();
    let mut a_out = vec![0.0; units];
    for j in 0..units {
        let mut z = 0.0;
        for i in 0..a_in.len() {
            z += w[i][j] * a_in[i];
        }
        z += b[j];
        a_out[j] = sigmoid_function(z.into()) as f32;
    }
    a_out
}

fn sequential_model(
    a_in: &[f32],
    w1: &[Vec<f32>],
    b1: &[f32],
    w2: &[Vec<f32>],
    b2: &[f32],
) -> Vec<f32> {
    let a_hidden = my_dense(a_in, w1, b1);
    my_dense(&a_hidden, w2, b2)
}

fn my_predict(
    x: &[Vec<f32>],
    w1: &[Vec<f32>],
    b1: &[f32],
    w2: &[Vec<f32>],
    b2: &[f32],
) -> Vec<Vec<f32>> {
    let mut predictions = Vec::new();
    for a_in in x {
        let a_out = sequential_model(a_in, w1, b1, w2, b2);
        predictions.push(a_out);
    }
    predictions
}

fn main() {
    let mut data = load_coffee_data(200);
    let norm_stats =
        NormalizationStats::new(vec![217.56155, 13.501258], vec![38.999992, 1.1572744]);
    apply_input_normalization(&mut data, &norm_stats);

    let mut raw_inputs = vec![vec![200.0, 13.9], vec![200.0, 17.0]];
    println!("Raw sample inputs: {:?}", raw_inputs);
    for input in raw_inputs.iter_mut() {
        normalize_single_input(input, &norm_stats);
    }
    println!("Normalized sample inputs: {:?}", raw_inputs);

    let w1 = vec![
        vec![-12.696792, 0.08674376, 18.024025],
        vec![0.065991946, -9.936261, 15.134285],
    ];
    let b1 = vec![-15.026951, -14.071334, 1.0949218];
    let w2 = vec![vec![-35.28394], vec![-34.293053], vec![-37.196335]];
    let b2 = vec![8.992294];

    let x_data: Vec<Vec<f32>> = data.iter().map(|d| d.input.clone()).collect();
    let y_true: Vec<Vec<f32>> = data.iter().map(|d| d.target.clone()).collect();

    let predict_data = my_predict(&x_data, &w1, &b1, &w2, &b2);

    let mut correct = 0;
    for (y_pred, y) in predict_data.iter().zip(y_true.iter()) {
        if (y_pred[0] - y[0]).abs() < 0.05 {
            correct += 1;
        }
    }

    println!(
        "Correct predictions (|diff| < 0.05): {} / {}",
        correct,
        y_true.len()
    );
    println!(
        "Accuracy: {:.2}%",
        (correct as f32 / y_true.len() as f32) * 100.0
    );
}
