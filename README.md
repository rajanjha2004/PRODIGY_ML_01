Task 
Implement a linear regression model to predict the prices of houses based on their square footage and the number of bedrooms and bathrooms.

Task Objective
The objective of this project is to develop a linear regression model that predicts house prices based on various features such as square footage, number of bedrooms, number of bathrooms, and other features( Optional)

Steps Taken to Satisfy the Requirements of the Task

1. Data Loading: The dataset was loaded from a CSV file, which contains various features relevant to house prices.

2. Feature Selection: Multiple relevant features were selected for model training. 

The features included:

--> GrLivArea: Above ground living area in square feet.
--> BedroomAbvGr: Number of bedrooms above ground.
--> FullBath: Number of full bathrooms.
--> OverallQual: Overall quality rating of the house.
--> YearBuilt: The year the house was built.
--> TotalBsmtSF: Total basement area in square feet.
--> GarageCars: Number of cars that can fit in the garage.
--> GarageArea: Area of the garage in square feet.
--> LotArea: Lot size in square feet.

3. Missing Values Handling: Rows with missing values were dropped from the training dataset. For the test dataset, missing values were filled with the mean of their respective features.

4. Data Visualization: Scatter plots were created to visualize the relationship between selected features (e.g., square footage, number of bedrooms) and house prices.

5. Data Splitting: The training data was split into training and test sets (80-20 split) to evaluate the model's performance effectively.

6. Feature Scaling: Standard scaling was applied to normalize the feature values, improving model performance.

7. Model Training: A linear regression model was trained on the scaled training data.

8. Model Evaluation: The model's performance was evaluated using Mean Squared Error (MSE) and R² Score on both training and testing datasets.

9. Predictions on Test Data: The trained model was used to predict house prices for the test dataset, which did not contain the target variable.

10. Submission Preparation: The predictions were formatted according to the submission guidelines, and a CSV file was created for submission.

11. Visualization of Results: A scatter plot was generated to compare actual vs. predicted house prices for the test set.

Conclusion
The implemented model demonstrates the capability to predict house prices based on selected features, showcasing the utility of linear regression in real estate price prediction. Further improvements could include feature engineering and the application of more advanced regression techniques.

