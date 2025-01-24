# Food-Demand-Forecasting
Overview

This project is designed to forecast food demand for a given dataset. By leveraging data preprocessing techniques and exploratory data analysis (EDA), it provides insights into factors influencing food demand and prepares the data for predictive modeling.

Features

Data loading and merging from multiple sources.

Exploratory data analysis to understand data distributions and relationships.

Log transformation of the target variable (num_orders) for normalization.

Technologies Used

Python

Pandas: For data manipulation and analysis.

NumPy: For numerical computations.

Seaborn: For data visualization.

Matplotlib: For plotting graphs.

Warnings: To suppress unwanted warnings during runtime.

Data Files

The script requires the following datasets:

sample_submission_hSlSoT6.csv: Sample submission file.

fulfilment_center_info.csv: Information about fulfillment centers.

meal_info.csv: Details about meals.

train.csv: Training dataset.

test_QoiMO9B.csv.xls: Testing dataset.

How to Use

Clone the repository:

git clone <repository_url>

Navigate to the project directory and ensure all required datasets are available in the specified paths.

Install the required Python libraries:

pip install -r requirements.txt

Run the script:

python FoodDemandForecasting.py

Outputs

Merged datasets combining all relevant information.

Visualizations of data distributions.

Log-transformed target variable prepared for further modeling.
