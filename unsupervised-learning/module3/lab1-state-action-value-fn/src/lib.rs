#[derive(Debug, Clone)]
pub struct StateValues {
    pub state: usize,
    pub q_left: f64,
    pub q_right: f64,
    pub v: f64,
}

pub fn compute_state_action_values(
    num_states: usize,
    terminal_left_reward: f64,
    terminal_right_reward: f64,
    gamma: f64,
) -> Vec<StateValues> {
    // Initialize values
    let mut values: Vec<StateValues> = (0..num_states)
        .map(|state| {
            let mut q_left = gamma.powi(state as i32) * terminal_left_reward;
            let mut q_right = gamma.powi((num_states - 1 - state) as i32) * terminal_right_reward;

            // Override for terminals
            if state == 0 {
                q_right = terminal_left_reward;
            } else if state == num_states - 1 {
                q_left = terminal_right_reward;
            }

            let v = q_left.max(q_right);

            StateValues {
                state,
                q_left,
                q_right,
                v,
            }
        })
        .collect();

    // Bellman updates until convergence
    let mut updated = true;
    while updated {
        updated = false;

        for i in 1..num_states - 1 {
            let max_left = values[i - 1].q_left.max(values[i - 1].q_right);
            let new_q_left = max_left * gamma;

            if values[i].q_left < new_q_left {
                values[i].q_left = new_q_left;
                values[i].v = values[i].v.max(new_q_left);
                updated = true;
            }

            let max_right = values[i + 1].q_left.max(values[i + 1].q_right);
            let new_q_right = max_right * gamma;

            if values[i].q_right < new_q_right {
                values[i].q_right = new_q_right;
                values[i].v = values[i].v.max(new_q_right);
                updated = true;
            }
        }
    }

    values
}
