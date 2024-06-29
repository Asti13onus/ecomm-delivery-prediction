Overview:
The aim of this project is to predict whether products from an international e-commerce company will reach customers on time or not. Additionally, the project analyzes various factors influencing product delivery and studies customer behavior. The company primarily sells electronic products.

Data Dictionary:
The dataset used for model building contains 10,999 observations of 12 variables, including:

| Variable             | Description                                             |
|----------------------|---------------------------------------------------------|
| ID                   | ID Number of Customers                                 |
| Warehouse_block      | The Company's Warehouse block (A, B, C, D, E)          |
| Mode_of_Shipment     | The mode of shipment (Ship, Flight, Road)              |
| Customer_care_calls  | Number of calls made for shipment inquiries            |
| Customer_rating      | Customer rating (1 - Lowest, 5 - Highest)              |
| Cost_of_the_Product  | Cost of the product in US Dollars                     |
| Prior_purchases      | Number of prior purchases                              |
| Product_importance   | Product importance categorization (low, medium, high)  |
| Gender               | Gender of customers (Male, Female)                    |
| Discount_offered     | Discount offered on specific products                  |
| Weight_in_gms        | Weight of the product in grams                         |
| Reached.on.Time_Y.N  | Target variable (1 - Product did not reach on time, 0 - Product reached on time) |

Conclusion:
The project's goal was to anticipate product delivery timeliness and investigate the elements that influence delivery and consumer behavior. Exploratory data analysis revealed that product weight and pricing had a substantial effect on delivery results. Products weighing between 2500 and 3500 grams and priced under $250 were more likely to arrive on time. Furthermore, the bulk of cargo began at Warehouse F and were delivered by ship, demonstrating the warehouse's closeness to a seaport.

Customer behavior has a huge influence on delivery timeliness. Delayed delivery were connected with an increase in customer care calls. Customers who had made more previous purchases, on the other hand, had greater rates of on-time delivery, indicating their devotion to the firm. Products with discounts between 0 and 10% were more likely to be delivered late, whereas those with discounts more than 10% had a higher likelihood of being delivered on time.

In terms of machine learning models, the decision tree classifier achieved the highest accuracy at 69%, outperforming other models. The random forest classifier and logistic regression achieved accuracies of 68% and 67%, respectively, while the K Nearest Neighbors model had the lowest accuracy at 65%.
