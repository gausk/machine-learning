use serde::Deserialize;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::preprocessing::categorical::{OneHotEncoder, OneHotEncoderParams};
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;
use smartcore::xgboost::XGRegressor;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
enum Sex {
    M = 0,
    F = 1,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
enum ChestPainType {
    Ata = 0,
    Nap = 1,
    Asy = 2,
    Ta = 3,
}

#[derive(Debug, Clone, Deserialize)]
enum Ecg {
    Normal = 0,
    ST = 1,
    #[serde(rename = "LVH")]
    Lvh = 2,
}

#[derive(Debug, Clone, Deserialize)]
enum Angina {
    Y = 0,
    N = 1,
}

#[derive(Debug, Clone, Deserialize)]
enum Slope {
    Up = 0,
    Flat = 1,
    Down = 2,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct Record {
    age: u32,
    sex: Sex,
    chest_pain_type: ChestPainType,
    #[serde(rename = "RestingBP")]
    resting_bp: u32,
    cholesterol: u32,
    #[serde(rename = "FastingBS")]
    fasting_bs: u8,
    #[serde(rename = "RestingECG")]
    resting_ecg: Ecg,
    #[serde(rename = "MaxHR")]
    max_hr: u32,
    exercise_angina: Angina,
    oldpeak: f32,
    #[serde(rename = "ST_Slope")]
    st_slope: Slope,
    heart_disease: u8,
}

impl Record {
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.age as f64,
            self.sex.clone() as u8 as f64,
            self.chest_pain_type.clone() as u8 as f64,
            self.resting_bp as f64,
            self.cholesterol as f64,
            self.fasting_bs as f64,
            self.resting_ecg.clone() as u8 as f64,
            self.max_hr as f64,
            self.exercise_angina.clone() as u8 as f64,
            self.oldpeak as f64,
            self.st_slope.clone() as u8 as f64,
        ]
    }
}

pub fn load_data() -> (DenseMatrix<f64>, Vec<u8>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/heart.csv");
    let mut reader = csv::Reader::from_path(path).unwrap();
    let mut input: Vec<Vec<f64>> = Vec::new();
    let mut output: Vec<u8> = Vec::new();
    for result in reader.deserialize::<Record>() {
        let record = result.unwrap();

        input.push(record.to_vec());
        output.push(record.heart_disease);
    }

    let data = DenseMatrix::from_2d_vec(&input).unwrap();
    let encoder =
        OneHotEncoder::fit(&data, OneHotEncoderParams::from_cat_idx(&[1, 2, 6, 8, 10])).unwrap();
    (encoder.transform(&data).unwrap(), output)
}

pub fn decision_tree_classifier(input: &DenseMatrix<f64>, output: &Vec<u8>) -> f64 {
    let tree = DecisionTreeClassifier::fit(input, output, Default::default()).unwrap();
    let predcited = tree.predict(input).unwrap();
    output
        .iter()
        .zip(predcited.iter())
        .filter(|&(&expected, &found)| expected == found)
        .count() as f64
        / output.len() as f64
}

pub fn xgboost(input: &DenseMatrix<f64>, output: &Vec<u8>) -> f64 {
    let tree = XGRegressor::fit(input, output, Default::default()).unwrap();
    let predcited = tree.predict(input).unwrap();
    output
        .iter()
        .zip(predcited.iter())
        .filter(|&(&expected, &found)| expected == found.round() as u8)
        .count() as f64
        / output.len() as f64
}

pub fn random_forest_classifier(input: &DenseMatrix<f64>, output: &Vec<u8>) -> f64 {
    let tree = RandomForestClassifier::fit(input, output, Default::default()).unwrap();
    let predcited = tree.predict(input).unwrap();
    output
        .iter()
        .zip(predcited.iter())
        .filter(|&(&expected, &found)| expected == found)
        .count() as f64
        / output.len() as f64
}
