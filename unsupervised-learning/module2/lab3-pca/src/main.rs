use linfa::DatasetBase;
use linfa::traits::{Fit, Predict};
use linfa_reduction::Pca;
use ndarray::array;

fn main() {
    println!("Welcome to Lab3 on Principal Component Analysis.");
    let data = array![
        [99.0, -1.0],
        [98.0, -1.0],
        [97.0, -2.0],
        [101.0, 1.0],
        [102.0, 1.0],
        [103.0, 2.0],
    ];
    let dataset = DatasetBase::from(data);
    println!("Data: {:?}", dataset);
    let embedding_2d = Pca::params(2).fit(&dataset).unwrap();
    let data_2d_red = embedding_2d.predict(&dataset);
    println!("Data 2d reduced: {:?}", data_2d_red);
    let data_2d_inverse = embedding_2d.inverse_transform(data_2d_red);
    println!("Data 2d inverse: {:?}", data_2d_inverse);

    let embedding_1d = Pca::params(1).fit(&dataset).unwrap();
    let data_1d = embedding_1d.predict(&dataset);
    println!("Data 1d reduced: {:?}", data_1d);
    let data_1d_inverse = embedding_1d.inverse_transform(data_1d);
    println!("Data 1d inverse: {:?}", data_1d_inverse);

    println!("\nTry PCA with new data\n");
    let data = array![
        [-0.83934975, -0.21160323],
        [0.67508491, 0.25113527],
        [-0.05495253, 0.36339613],
        [-0.57524042, 0.24450324],
        [0.58468572, 0.95337657],
        [0.5663363, 0.07555096],
        [-0.50228538, -0.65749982],
        [-0.14075593, 0.02713815],
        [0.2587186, -0.26890678],
        [0.02775847, -0.77709049]
    ];
    let dataset = DatasetBase::from(data);
    println!("Data: {:?}", dataset);
    let embedding_2d = Pca::params(2).fit(&dataset).unwrap();
    let data_2d_red = embedding_2d.predict(&dataset);
    println!("Data 2d reduced: {:?}", data_2d_red);
    let data_2d_inverse = embedding_2d.inverse_transform(data_2d_red);
    println!("Data 2d inverse: {:?}", data_2d_inverse);

    let embedding_1d = Pca::params(1).fit(&dataset).unwrap();
    let data_1d = embedding_1d.predict(&dataset);
    println!("Data 1d reduced: {:?}", data_1d);
    let data_1d_inverse = embedding_1d.inverse_transform(data_1d);
    println!("Data 1d inverse: {:?}", data_1d_inverse);
}
