use lab1_decision_tree::{build_tree_recursive, feature_selection};

fn main() {
    println!("Welcome to Lab1 on DecisionTree!");

    let x_train = Vec::from([
        vec![1, 1, 1],
        vec![0, 0, 1],
        vec![0, 1, 0],
        vec![1, 0, 1],
        vec![1, 1, 1],
        vec![1, 1, 0],
        vec![0, 0, 0],
        vec![1, 1, 0],
        vec![0, 1, 0],
        vec![0, 1, 0],
    ]);

    let y_train = Vec::from([1, 1, 0, 0, 1, 1, 0, 1, 0, 0]);

    feature_selection(&x_train, &y_train);

    build_tree_recursive(&x_train, &y_train, 3);
}
