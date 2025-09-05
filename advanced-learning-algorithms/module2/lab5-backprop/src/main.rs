use ahash::HashMap;
use symbolica::{atom::AtomCore, parse, symbol};

fn main() {
    println!("Welcome to lab 5");

    println!("Testing forward propagation");
    // a = 2 + 3w, j = a^2
    let w = 3f64;
    let a = 2.0 + 3.0 * w;
    let j = a.powi(2);
    println!("a = {a}, J = {j}");

    let a_epsilon = a + 0.001;
    let j_epsilon = a_epsilon.powi(2);
    let k = (j_epsilon - j) / 0.001;
    println!("j = {j}, j_epsilon = {j_epsilon}, dJ_da ~= k = {k}");

    let j_s = parse!("a^2");
    let a_s = parse!("2 + 3w");
    let dj_da = j_s.derivative(symbol!("a"));
    println!("dj_da = {dj_da} where j = {j_s}");
    let mut const_map = HashMap::default();
    let var_a = parse!("a");
    const_map.insert(var_a, a);
    let dj_da_val = dj_da
        .evaluate(|r| r.to_f64(), &const_map, &HashMap::default())
        .unwrap();
    println!("Evaluated dj_da = {dj_da_val} where a = {a}");

    let w_epsilon = w + 0.001;
    let a_epsilon = 2.0 + 3.0 * w_epsilon;
    let k = (a_epsilon - a) / 0.001;
    println!("a = {a}, a_epsilon = {a_epsilon}, da_dw ~= k = {k} ");
    let da_dw = a_s.derivative(symbol!("w"));
    println!("da_dw = {da_dw} where a = {a_s}");
    let da_dw_val = da_dw
        .evaluate(|r| r.to_f64(), &const_map, &HashMap::default())
        .unwrap();
    println!("Evaluated da_dw = {da_dw_val} where w = {w}");

    let dj_dw = dj_da_val * da_dw_val;
    println!("Evaluated dj_dw = {dj_dw} where a = {a} and w = {w}");

    let w_epsilon = w + 0.001;
    let a_epsilon = 2.0 + 3.0 * w_epsilon;
    let j_epsilon = a_epsilon.powi(2);
    let k = (j_epsilon - j) / 0.001;
    println!("j = {j}, j_epsilon = {j_epsilon}, dj_dw ~= k = {k}");

    println!("Testing backward propagation");
    // Inputs and parameters
    let x = 2f64;
    let w = -2f64;
    let b = 8f64;
    let y = 1f64;

    // calculate per step values
    let c = w * x;
    let a = c + b;
    let d = a - y;
    let j = d.powi(2) / 2.0;
    println!("j={j}, d={d}, a={a}, c={c}");

    let d_epsilon = d + 0.001;
    let j_epsilon = d_epsilon.powi(2) / 2.0;
    let k = (j_epsilon - j) / 0.001;
    println!("j = {j}, j_epsilon = {j_epsilon}, dJ_dd ~= k = {k} ");

    let j_s = parse!("d^2/2");
    let dj_dd = j_s.derivative(symbol!("d"));
    println!("dj_dd = {dj_dd} where j = {j_s}");
    const_map.insert(parse!("d"), d);
    let dj_dd_val = dj_dd
        .evaluate(|r| r.to_f64(), &const_map, &HashMap::default())
        .unwrap();
    println!("dj_dd = {dj_dd_val} where d = {d}");

    let a_epsilon = a + 0.001;
    let d_epsilon = a_epsilon - y;
    let k = (d_epsilon - d) / 0.001;
    println!("d = {d}, d_epsilon = {d_epsilon}, dd_da ~= k = {k} ");

    let d_s = parse!("a - y");
    let dd_da = d_s.derivative(symbol!("a"));
    println!("dd_da = {dd_da} where d = {d_s}");
    const_map.insert(parse!("a"), a);
    const_map.insert(parse!("k"), k);
    let dd_da_val = dd_da
        .evaluate(|r| r.to_f64(), &const_map, &HashMap::default())
        .unwrap();
    println!("dd_da = {dd_da_val} where a = {a}");

    let dj_da_val = dj_dd_val * dd_da_val;
    println!("dj_da_val = {dj_da_val} where a = {a}");

    let a_epsilon = a + 0.001;
    let d_epsilon = a_epsilon - y;
    let j_epsilon = d_epsilon.powi(2) / 2.0;
    let k = (j_epsilon - j) / 0.001;
    println!("j = {j}, j_epsilon = {j_epsilon}, dJ_da ~= k = {k}");

    let a_s = parse!("c+b");
    let da_db = a_s.derivative(symbol!("b"));
    println!("da_db = {da_db} where a = {a_s}");
    let da_dc = a_s.derivative(symbol!("c"));
    println!("da_dc = {da_dc} where a = {a_s}");
    const_map.insert(parse!("c"), c);
    const_map.insert(parse!("b"), b);
    let da_db_val = da_db
        .evaluate(|r| r.to_f64(), &const_map, &HashMap::default())
        .unwrap();
    let da_dc_val = da_dc
        .evaluate(|r| r.to_f64(), &const_map, &HashMap::default())
        .unwrap();
    println!("da_db = {da_db_val} where b = {b}");
    println!("da_dc = {da_dc_val} where c = {c}");

    let dj_dc_val = da_dc_val * dj_da_val;
    let dj_db_val = da_db_val * dj_da_val;
    println!("dj_dc = {dj_dc_val},  dj_db = {dj_db_val}");

    let c_s = parse!("w*x");
    let dc_dw = c_s.derivative(symbol!("w"));
    println!("dc_dw = {dc_dw} where c = {c_s}");
    const_map.insert(parse!("w"), w);
    const_map.insert(parse!("x"), x);
    let dc_dw_val = dc_dw
        .evaluate(|r| r.to_f64(), &const_map, &HashMap::default())
        .unwrap();
    println!("dc_dw = {dc_dw_val} where w = {w}");

    let dj_dw_val = dc_dw_val * dj_dc_val;
    println!("dj_dw = {dj_dw_val} where w = {w}");

    let j_epsilon = ((w + 0.001) * x + b - y).powi(2) / 2.0;
    let k = (j_epsilon - j) / 0.001;
    println!("j = {j}, j_epsilon = {j_epsilon}, dJ_dw ~= k = {k}");
}
