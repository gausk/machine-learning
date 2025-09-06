pub fn entropy(p: f64) -> f64 {
    if p == 0.0 || p == 1.0 {
        return 0f64;
    }
    -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
}

/// Given a dataset and a index feature, return two lists for the two split nodes, the left node has the animals that have
/// that feature = 1 and the right node those that have the feature = 0
pub fn split_indices(
    x: &[Vec<i32>],
    node_indices: &[usize],
    index: usize,
) -> (Vec<usize>, Vec<usize>) {
    let mut left = vec![];
    let mut right = vec![];
    for &i in node_indices {
        if x[i][index] == 1 {
            left.push(i);
        } else {
            right.push(i);
        }
    }
    (left, right)
}

/// This function takes the split dataset, the indices we chose to split and returns the weighted entropy.
pub fn weighted_entropy(y: &[i32], left_indices: &[usize], right_indices: &[usize]) -> f64 {
    let total_size = left_indices.len() + right_indices.len();
    let w_left = left_indices.len() as f64 / total_size as f64;
    let w_right = right_indices.len() as f64 / total_size as f64;

    let positive_left_prob =
        left_indices.iter().map(|&i| y[i]).sum::<i32>() as f64 / left_indices.len() as f64;
    let positive_right_prob =
        right_indices.iter().map(|&i| y[i]).sum::<i32>() as f64 / right_indices.len() as f64;

    w_left * entropy(positive_left_prob) + w_right * entropy(positive_right_prob)
}

pub fn information_gain(y: &[i32], left_indices: &[usize], right_indices: &[usize]) -> f64 {
    let total_size = left_indices.len() + right_indices.len();
    let p_node = (left_indices.iter().map(|&i| y[i]).sum::<i32>()
        + right_indices.iter().map(|&i| y[i]).sum::<i32>()) as f64
        / total_size as f64;
    let h_node = entropy(p_node);

    let weighted_entropy = weighted_entropy(y, left_indices, right_indices);
    h_node - weighted_entropy
}

pub fn feature_selection(x: &[Vec<i32>], y: &[i32]) -> (usize, Vec<usize>, Vec<usize>) {
    feature_selection_with_indices(x, y, &(0..x.len()).collect::<Vec<usize>>(), true)
}

pub fn feature_selection_with_indices(
    x: &[Vec<i32>],
    y: &[i32],
    node_indices: &[usize],
    is_verbose: bool,
) -> (usize, Vec<usize>, Vec<usize>) {
    let features = x[0].len();
    let mut idx = 0;
    let mut left_indices = vec![];
    let mut right_indices = vec![];
    let mut max_gain = 0.0;
    for i in 0..features {
        let (left, right) = split_indices(x, node_indices, i);
        let gain = information_gain(y, &left, &right);
        if is_verbose {
            println!(
                "Information gain if we split the root node using this feature {i} is: {gain}"
            );
            println!("Left: {left:?}, Right: {right:?}");
        }
        if gain > max_gain {
            max_gain = gain;
            idx = i;
            left_indices = left;
            right_indices = right;
        }
    }
    (idx, left_indices, right_indices)
}

pub fn build_tree_recursive_with_indices(
    x: &[Vec<i32>],
    y: &[i32],
    max_depth: usize,
    current_depth: usize,
    indices: &[usize],
) {
    if current_depth >= max_depth || x.len() <= 1 {
        return;
    }

    if indices.iter().all(|&i| y[i] == 1) || indices.iter().all(|&i| y[i] == 0) {
        println!("Depth: {current_depth} All elements in same category: {indices:?}");
        return;
    }

    let (idx, left_indices, right_indices) = feature_selection_with_indices(x, y, indices, false);
    println!(
        "Depth: {current_depth} Selected feature is {idx}, left: {left_indices:?}, right: {right_indices:?}, total_elements: {}",
        indices.len()
    );
    build_tree_recursive_with_indices(x, y, max_depth, current_depth + 1, &left_indices);
    build_tree_recursive_with_indices(x, y, max_depth, current_depth + 1, &right_indices);
}

pub fn build_tree_recursive(x: &[Vec<i32>], y: &[i32], max_depth: usize) {
    println!("Build Decision Tree");
    build_tree_recursive_with_indices(x, y, max_depth, 0, &(0..x.len()).collect::<Vec<_>>());
}
