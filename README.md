# Heart Disease Prediction with Machine Learning

This project implements a machine learning model to predict the presence of heart disease based on medical attributes. It utilizes a Decision Tree Classifier and includes data exploration, preprocessing, and evaluation steps.

## Overview

Heart disease is a leading cause of death globally. Early and accurate prediction of heart disease can significantly improve patient outcomes. This project aims to develop a predictive model using machine learning techniques to identify individuals at risk of heart disease based on various medical features.

## Dataset

The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/). It contains various medical attributes of patients, such as age, sex, chest pain type, blood pressure, cholesterol levels, and heart rate, along with a target variable indicating the presence or absence of heart disease.

**Data Source:** [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets) (Replace with the specific Kaggle dataset URL if you know it!)

**Dataset Description:**  (You should customize this based on the specific Kaggle dataset you used.)

*   **Features:** The dataset includes the following features:
    *   age: Patient's age
    *   sex: Patient's sex (1 = male, 0 = female)
    *   cp: Chest pain type (various categories)
    *   trestbps: Resting blood pressure
    *   chol: Serum cholesterol (mg/dl)
    *   fbs: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    *   restecg: Resting electrocardiographic results (various categories)
    *   thalach: Maximum heart rate achieved
    *   exang: Exercise induced angina (1 = yes, 0 = no)
    *   oldpeak: ST depression induced by exercise relative to rest
    *   slope: The slope of the peak exercise ST segment
    *   ca: Number of major vessels (0-3) colored by fluoroscopy
    *   thal: Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
    *   AHD: Target variable - Heart Disease presence or absence (1 = yes, 0 = no)


**Explanation:**
* You may rename `Heart_Disease_Prediction.ipynb` to a more descriptive name to fit your purposes

## Libraries Used

*   **pandas:** For data manipulation and analysis.
*   **numpy:** For numerical computations.
*   **matplotlib:** For data visualization.
*   **seaborn:** For enhanced data visualization.
*   **scikit-learn (sklearn):** For machine learning tasks (model training, evaluation, etc.).
*   **imblearn:** For handling imbalanced datasets using SMOTE (Synthetic Minority Oversampling Technique).

## Implementation

The Jupyter Notebook (`Heart_Disease_Prediction.ipynb`) contains the following steps:

1.  **Data Loading and Exploration:**
    *   Loading the dataset using pandas.
    *   Displaying summary statistics and information about the data.
    *   Checking for missing values.

2.  **Exploratory Data Analysis (EDA):**
    *   Visualizing data distributions using histograms and pairplots.
    *   Creating a correlation heatmap to identify relationships between features.
    *   Analyzing the distribution of the target variable (heart disease presence).

3.  **Data Preprocessing:**
    *   Splitting the data into training and testing sets.
    *   Handling class imbalance using SMOTE (Synthetic Minority Oversampling Technique) to balance the target variable.

4.  **Model Training and Evaluation:**
    *   Training a Decision Tree Classifier.
    *   Evaluating the model's performance using metrics such as accuracy, precision, recall, and F1-score.
    *   Performing Leave-One-Out Cross-Validation (LOOCV).

5.  **Results and Conclusion:** Summarizing the findings and discussing the model's performance.

## Running the Code

1.  **Install Dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn imblearn
    ```

2.  **Open the Jupyter Notebook:**
    ```bash
    jupyter notebook Heart_Disease_Prediction.ipynb
    ```

3.  **Run the Notebook:** Execute the cells in the notebook sequentially to perform the analysis and train the model.

## Results

The project achieved the following results (These are based on your notebook, you can adjust them):

*   Decision Tree Classifier achieved an accuracy of approximately 72% on regular training, 79% on LOOCV and 89% when applied with the SMOTE resampling.
*   SMOTE oversampling technique improved the model's ability to correctly identify both positive and negative cases of heart disease.
*   Visualizations helped provide valuable insights into the data and relationships between features.


## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please feel free to submit a pull request.

## License

[MIT License](LICENSE) 
