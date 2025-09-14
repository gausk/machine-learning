use lab1_state_action_value_fn::compute_state_action_values;

fn main() {
    let values = compute_state_action_values(6, 100.0, 40.0, 0.5);

    println!("state | Q(left) | Q(right) | V(state)");
    for v in values {
        println!(
            "{:>5} | {:>7.3} | {:>8.3} | {:>8.3}",
            v.state, v.q_left, v.q_right, v.v
        );
    }
}
