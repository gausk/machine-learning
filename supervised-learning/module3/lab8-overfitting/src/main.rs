fn main() {
    println!(
        "Overfiiting happens when a model learns the training data too well, including its noise and outliers, leading to poor generalization on new, unseen data."
    );
    println!(
        "This is often due to an overly complex model relative to the amount of training data available."
    );
    println!(
        "For example, using a high-degree polynomial to fit a small dataset can result in a model that performs well on training data but poorly on test data."
    );
    println!(
        "Using higher degrees of polynomial features can lead to overfitting, as the model may fit the training data too closely."
    );

    println!(
        "To mitigate overfitting, techniques such as cross-validation, regularization, pruning, early stopping, and using more training data can be employed."
    );
}
