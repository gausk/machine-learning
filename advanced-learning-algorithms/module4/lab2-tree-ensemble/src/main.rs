use lab2_tree_ensemble::{decision_tree_classifier, load_data, random_forest_classifier, xgboost};
use smartcore::linalg::basic::arrays::Array;

fn main() {
    println!("Welcome to Lab2 on Tree ensemble.");
    let (input, output) = load_data();
    println!("Input data shape: {:?}", input.shape());
    println!("Output data shape: {:?}", output.shape());

    println!("Running Decision Tree Classifier");
    let accuracy = decision_tree_classifier(&input, &output);
    println!("Decision Tree Classifier accuracy: {}", accuracy);

    println!("Running Random Forest Classifier");
    let accuracy = random_forest_classifier(&input, &output);
    println!("Random Forest Classifier accuracy: {}", accuracy);

    println!("Running XGBoost Classifier");
    let accuracy = xgboost(&input, &output);
    println!("XGBoost Classifier accuracy: {}", accuracy);
}
