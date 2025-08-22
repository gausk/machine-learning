use lab2_sigmoid::compute_cost_logistic;
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
    let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    let w_array1 = array![1.0, 1.0];
    let b_1 = -3.0;
    let cost1 = compute_cost_logistic(&x, &y, &w_array1, b_1);
    println!("Cost with w=[1,1] and b=-3: {}", cost1);

    let w_array2 = array![1.0, 1.0];
    let b_2 = -4.0;
    let cost2 = compute_cost_logistic(&x, &y, &w_array2, b_2);
    println!("Cost with w=[1,1] and b=-4: {}", cost2);
}
