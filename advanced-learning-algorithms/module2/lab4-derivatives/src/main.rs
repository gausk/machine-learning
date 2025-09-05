use ahash::HashMap;
use symbolica::{atom::AtomCore, parse, symbol};

fn main() {
    println!("Welcome to lab 4 on Derivatives");

    // w^2
    let j = (3.0_f64).powi(2);
    let j_epsilon = (3.0_f64 + 0.001).powi(2);
    let dj_dw = (j_epsilon - j) / 0.001;
    println!("J: {j}, J_epsilon: {j_epsilon}, dj_dw: {dj_dw:.6}");

    let j = (3.0_f64).powi(2);
    let j_epsilon = (3.0_f64 + 0.000000001).powi(2);
    let dj_dw = (j_epsilon - j) / 0.000000001;
    println!("J: {j}, J_epsilon: {j_epsilon}, dj_dw: {dj_dw}");

    let w = parse!("w");
    let input = parse!("w^2");
    let derivative = input.derivative(symbol!("w"));
    println!("d/dw {} = {}:", input, derivative);
    let mut var_map = HashMap::default();
    var_map.insert(w.clone(), 3.0);
    let dj_dw = derivative
        .evaluate(|r| r.to_f64(), &var_map, &HashMap::default())
        .unwrap();
    println!("Evaluated dj/dw: {}", dj_dw);

    // 2*w
    let j = 2.0 * 3.0_f64;
    let j_epsilon = 2.0 * (3.0_f64 + 0.001);
    let dj_dw = (j_epsilon - j) / 0.001;
    println!("J: {j}, J_epsilon: {j_epsilon}, dj_dw: {dj_dw}");

    let input = parse!("2*w");
    let derivative = input.derivative(symbol!("w"));
    println!("d/dw {} = {}:", input, derivative);
    let dj_dw = derivative
        .evaluate(|r| r.to_f64(), &var_map, &HashMap::default())
        .unwrap();
    println!("Evaluated dj/dw: {}", dj_dw);

    // w^3
    let j = 2.0_f64.powi(3);
    let j_epsilon = (2.0_f64 + 0.001).powi(3);
    let dj_dw = (j_epsilon - j) / 0.001;
    println!("J: {j}, J_epsilon: {j_epsilon}, dj_dw: {dj_dw}");

    let input = parse!("w^3");
    let derivative = input.derivative(symbol!("w"));
    println!("d/dw {} = {}:", input, derivative);
    var_map.insert(w.clone(), 2.0);
    let dj_dw = derivative
        .evaluate(|r| r.to_f64(), &var_map, &HashMap::default())
        .unwrap();
    println!("Evaluated dj/dw: {}", dj_dw);

    // 1/w
    let j = 2.0_f64.powi(-1);
    let j_epsilon = (2.0_f64 + 0.001).powi(-1);
    let dj_dw = (j_epsilon - j) / 0.001;
    println!("J: {j}, J_epsilon: {j_epsilon}, dj_dw: {dj_dw}");

    let input = parse!("1/w");
    let derivative = input.derivative(symbol!("w"));
    println!("d/dw {} = {}:", input, derivative);
    let dj_dw = derivative
        .evaluate(|r| r.to_f64(), &var_map, &HashMap::default())
        .unwrap();
    println!("Evaluated dj/dw: {}", dj_dw);

    // 1/w^2
    let j = 4.0_f64.powi(-2);
    let j_epsilon = (4.0_f64 + 0.001).powi(-2);
    let dj_dw = (j_epsilon - j) / 0.001;
    println!("J: {j}, J_epsilon: {j_epsilon}, dj_dw: {dj_dw}");

    let input = parse!("1/w^2");
    let derivative = input.derivative(symbol!("w"));
    println!("d/dw {} = {}:", input, derivative);
    var_map.insert(w.clone(), 4.0);
    let dj_dw = derivative
        .evaluate(|r| r.to_f64(), &var_map, &HashMap::default())
        .unwrap();
    println!("Evaluated dj/dw: {}", dj_dw);
}
