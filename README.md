# House Price Prediction Using Machine Learning

## Project Overview
This project uses machine learning to predict house prices based on property features such as area, number of bedrooms, bathrooms, stories, parking, and furnishing status.

The project includes data analysis, data preprocessing, model training, model evaluation, and a Streamlit web application for making predictions.

## Dataset
The dataset is stored in `data/Housing.csv`.

It contains 545 rows and 13 columns, including the target variable `price` and 12 property features.

## Machine Learning Models
I trained and compared the following models:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Tuned Random Forest Regressor

Linear Regression achieved the best test-set performance among the models tested.

## Evaluation Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 970,043.4 | 1,324,507.0 | 0.653 |
| Decision Tree | 1,222,334.9 | 1,659,971.2 | 0.455 |
| Random Forest | 1,026,432.1 | 1,403,563.8 | 0.610 |
| Tuned Random Forest | 1,026,161.3 | 1,407,738.3 | 0.608 |

## Web Application
The project includes a Streamlit app called **Noir Estate | Property Valuation**. Users can enter property details and receive a price prediction from the saved machine learning model.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit
- Joblib

## How to Run the Project

1. Clone or download this repository.
2. Open a terminal in the project folder.
3. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the web application:

   ```bash
   python -m streamlit run app.py
   ```

5. Open the local URL shown in the terminal.

## Project Structure

```text
House-Price-Prediction/
├── app.py
├── main.py
├── house_price_model.pkl
├── requirements.txt
├── README.md
└── data/
    └── Housing.csv
```

## Future Improvements
- Explore additional regression models.
- Improve model performance through further experimentation.
- Deploy the Streamlit app online.

## Author
Add your name and GitHub profile here.
