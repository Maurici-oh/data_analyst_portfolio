# 🛠️ CSV Data Preprocessing App

This Streamlit app is designed to perform comprehensive data preprocessing for machine learning workflows. It handles tasks such as missing value imputation, encoding categorical variables, feature scaling, and outlier detection. The tool provides an interactive interface for users to upload datasets, apply transformations, and export the cleaned data for modeling. The project emphasizes automation, reproducibility, and ease of use for both beginners and data professionals.

## 🔗 WebApp URL
[mra-car-price-predictor-app.streamlit.app/](https://csvdatapreprocessingapp-mra11.streamlit.app/)

## 📊 Project Overview
**Goal**: Make dataset preprocessing more visual and user-friendly for both beginners and data professionals.

The application is divided in 10 sections (tabs):

**1. Data Exploration**
* Dataset Display: take a first glimpse of the uploaded dataset
* Dataset Description: take a deeper look into the dataset statistics
* Rename Column(s)

**2. Data Cleaning**
* Handle Missing Values: choose from several options for handling missing values in the dataset
    * Fill with Mean
    * Fill with Median
    * Fill with Meode
    * Fill with Custom Value
    * Drop Rows
* Remove Duplicates
    * Remove all duplicates
    * Remove all but first occurrence
* Data type convertion: convert columns from one datatype to another
    * int
    * float
    * str
    * datetime
    * category
    * bool
* Drop Columns
* Remove Outliers: includes three methods for outliers removal
    * IQR method
    * Top/bottom percentage
    * Isolation Forest method

**3. Exploratory Data Analysis (EDA)**
* Statistics summary table 
* Data visualization tools like:
    * Histogram
    * KDE plot
    * Box plot
    * Violin plot
    * Scatter plot
    * Heatmap
    * Line plot

**4. Data Splitting**
* Train-Test split
* Train-Validation-Test Split

**5. Data Transformation**
* Power Transformation: includes three transformation methods
    * Log Transformation
    * Box-Cox Transformation
    * Yeo-Johnson Transformation
* Data Scaling: includes three scaling methods for data standardization
    * Standard Scaler (Z-score)
    * MinMax Scaler (0-1 range)
    * Robust Scaler
* Categorical Variables Encoding: includes three encoding methods
    * Label Encoding
    * Ordinal Encoding
    * One-Hot Encoding
    * Target Encoding

**6. Dimensionality Reduction**
* Variance Threshold
* Principal Component Analysis (PCA)

**To be developed**
* **7.** Feature Engineering
* **8.** Binarization

**9. Data Export**
* Download the final preprocessed dataset

**10. Operations History**
* A record of all preprocessing steps applied to the dataset.


## 🛠️ Tech Stack

**Python**  
**Pandas**  
**Numpy**  
**Scikit-learn**   
**Streamlit**  
**Plotly**
**Scipy**
**uuid**
**copy**


## ⚙️ How to Run Locally

**1.** Clone the repository:
```bash
git clone https://github.com/Maurici-oh/csv_data_preprocessing_app.git
cd csv_data_preprocessing_app/
```
**2.** Install dependencies:

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements file.

```bash
pip install -r requirements.txt
```
**3.** Run the Streamlit app:

```bash
cd csv_data_preprocessing_app/
streamlit run app.py
```
**4.** Open the app in your browser at: `http://localhost:8501`

## 🧾 Folder Structure
<pre>
csv_data_preprocessing_app/  
├── .streamlit/
│   └── config.toml   
│   
├── app.py
├── requirements.txt
├── LICENSE
└── README.md
</pre>

**Note**: The `.streamlit/` folder contains the `config.toml` file, which disables the app’s responsiveness to the browser’s theme and forces it to use the dark theme.

## 📈 Example Output

* The output will be a .csv file that the user names prior to downloading.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or pull requests to improve the app.

## 📄 License

This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License.




