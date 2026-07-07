# PROJECT REQUIREMENTS

Academic task overview and coverage rubric for the Data Science & Marketing Analytics term paper.

Use this document to evaluate whether the code, outputs, and written report cover the lecturer's assignment.

## 1. Project context

In this term paper, the student acts as a customer intelligence analyst for a marketing research company.

The client is a restaurant owner or restaurant investor who wants to understand what drives customer satisfaction in restaurants.

The project uses Yelp data to analyze restaurant performance and customer behavior. Yelp provides data about businesses, reviews, users, check-ins, tips, and photos. These data can be used to create a structured analytical dataset, explore restaurant-related key performance indicators, and develop machine-learning models that predict customer satisfaction.

The project combines three perspectives:

1. Business perspective: What can restaurant owners or investors learn from Yelp customer behavior?
2. Data science perspective: How can Yelp data be transformed into a structured dataset suitable for analysis?
3. Marketing analytics perspective: Which factors are associated with satisfied or dissatisfied restaurant customers?

The final result should not only be technically correct, but also useful for managerial decision-making.

## 2. Core business problem

Restaurant owners and investors want to know which factors influence customer satisfaction.

Customer satisfaction may be affected by:

- restaurant attributes
- review behavior
- user characteristics
- visitor activity
- online engagement
- weather conditions
- socio-economic context
- other location-specific or temporal factors

The central business question is:

**Which restaurant, customer, review, engagement, and contextual factors help explain and predict customer satisfaction in restaurants?**

The goal is to provide insights that can help restaurant owners improve customer experience and help investors evaluate restaurant performance.

## 3. Main analytical objective

The main objective is to create a structured restaurant dataset from the Yelp database and use it to predict customer satisfaction.

The task has two major parts.

### 3.1 Dataset construction and preparation

The project must:

- select restaurant-related observations
- combine relevant Yelp tables
- define key performance indicators
- create a target variable for customer satisfaction
- clean and transform the data
- prepare the data for exploratory analysis
- prepare the data for machine learning

### 3.2 Machine-learning analysis

The project must:

- train different machine-learning models
- compare predictive performance
- identify the best model
- interpret the model results
- translate the findings into recommendations for restaurant owners or investors

The main prediction task is:

**Predict whether a restaurant customer is satisfied or not satisfied.**

## 4. Recommended unit of analysis

A central methodological decision is the unit of analysis.

The Yelp tables have different natural units:

| Table | Natural unit of observation |
|---|---|
| Business table | One business |
| Review table | One review |
| User table | One user |
| Check-in table | One business with many check-in timestamps |
| Tip table | One tip |
| Photo table | One photo |

Because customer satisfaction is most directly expressed in individual reviews, the recommended unit of analysis is the review level.

That means:

**One row in the final dataset represents one review written by one user for one restaurant at one point in time.**

This structure is suitable because the target variable can be derived from the review rating.

The different table structures must be handled carefully. Business-level, user-level, check-in-level, tip-level, and photo-level data cannot simply be joined without thought. If tables with many observations per business are joined directly, the number of rows may be artificially multiplied.

Information from other tables should usually be aggregated before joining.

Examples:

- aggregate check-ins per business
- aggregate number of tips per business
- aggregate number of photos per business
- aggregate photo labels per business
- join user characteristics by `user_id`
- join business characteristics by `business_id`

This avoids duplicated observations and ensures that the final dataset has a clear and consistent structure.

## 5. Project scope

The project focuses on businesses in the restaurant category.

A business should be included if its category field contains restaurant-related labels. The cleanest rule is:

**A business is classified as a restaurant if the `categories` field contains the term `Restaurants`.**

This rule should be documented clearly in the paper.

The project should avoid mixing restaurants with unrelated businesses such as retail stores, salons, car repair shops, or hotels.

## 6. Data source

The project is based on the Yelp dataset provided through the university PostgreSQL database.

The database contains the following relevant tables:

1. `business`
2. `review`
3. `user`
4. `checkin`
5. `tip`
6. `photo`

The original data are stored in JSON-style tables in the `public` schema. A transformed 25 percent sample of review-related data is also available in the `public3` schema.

The student may decide which schema to use. The choice should be justified.

A practical recommendation is:

- use the 25 percent transformed schema for initial exploration and prototyping
- use the full schema if computationally feasible and if the additional data improve the analysis

## 7. Key performance indicators

The assignment requires the creation of relevant KPIs for restaurants. These KPIs should describe restaurant performance, customer satisfaction, and customer engagement.

Possible KPIs include:

| KPI | Possible definition | Interpretation |
|---|---|---|
| Positive rating | Review stars >= 4 | Direct satisfaction indicator |
| Positive sentiment | Review text has positive sentiment | Text-based satisfaction indicator |
| Check-in behavior | Whether or how often customers check in | Visitor activity indicator |
| Check-in intensity | Number of check-ins per restaurant | Restaurant popularity / customer traffic |
| Review volume | Number of reviews per restaurant | Visibility and engagement |
| Average restaurant rating | Mean stars per restaurant | Overall performance |
| Rating above local median | Rating higher than local restaurant median | Relative local performance |
| Tip volume | Number of tips per restaurant | Customer interaction |
| Photo volume | Number of photos per restaurant | Visual engagement |
| Useful review votes | Number of useful votes | Review relevance or influence |

The exact KPI choice is up to the student. However, the chosen KPI or target variable must be clearly defined and justified.

For the machine-learning task, the most suitable main KPI is:

**Customer satisfaction as a binary yes/no variable.**

## 8. Target variable

The target variable should represent whether a customer is satisfied or not satisfied.

A clear and defensible definition is:

```text
satisfied = 1 if review_stars >= 4
satisfied = 0 if review_stars <= 3
```

This is recommended because Yelp star ratings are standardized and directly reflect customer evaluation.

Alternative target definitions are possible:

```text
satisfied = 1 if the review rating is above the median rating of restaurants in the same city or area
satisfied = 0 otherwise
```

or:

```text
satisfied = 1 if the review text sentiment is positive
satisfied = 0 otherwise
```

The first option, based on stars >= 4, is the most practical and easiest to explain. The other options can be used as robustness checks or extensions.

For this repository, the intended modelling setup is structured-feature prediction with the star-based target. Review text itself should not be used as a model input unless the user explicitly changes the project scope.

## 9. Predictor variables

The final dataset should contain variables that may help explain or predict customer satisfaction.

### 9.1 Restaurant-level variables

Possible restaurant-level predictors include:

- city
- state
- postal code
- latitude and longitude
- restaurant category
- cuisine type
- average star rating, if used only in a leakage-safe or descriptive way
- review count
- open or closed status
- price range
- takeout availability
- delivery availability
- reservation availability
- outdoor seating
- parking availability
- alcohol availability
- Wi-Fi availability
- ambience
- noise level
- opening hours
- number of photos
- number of tips
- number of check-ins

These variables describe the restaurant's characteristics, popularity, and service environment.

Important note: rating-derived variables such as business average stars may create leakage if used directly in the predictive model. They can be used descriptively or excluded from modelling.

### 9.2 Review-level variables

Possible review-level predictors include:

- review date
- review year
- review month
- day of week
- weekend indicator
- season
- review text length
- number of words in the review
- useful votes, if used only descriptively or excluded from leakage-safe modelling
- funny votes, if used only descriptively or excluded from leakage-safe modelling
- cool votes, if used only descriptively or excluded from leakage-safe modelling

Important methodological note:

If satisfaction is defined using the star rating, review text sentiment may be very close to the target variable. Review vote counts also happen after the review is posted and may create look-ahead leakage.

A leakage-safe structured-feature model should avoid:

- `review_text`
- review sentiment derived from `review_text`
- `review_useful`
- `review_funny`
- `review_cool`
- `review_stars` as an input feature

`review_text_length` may be acceptable as a structured proxy.

### 9.3 User-level variables

Possible user-level predictors include:

- user review count
- user account age
- number of friends
- number of fans
- elite status
- useful/funny/cool activity
- compliments received
- general activity level

These variables capture reviewer behavior. Some users may generally give higher ratings, while others may be more critical.

Important note: user average star rating is rating-derived and may create leakage or target contamination. It should be excluded from the predictive model unless there is a strong, time-safe justification.

### 9.4 Check-in variables

Check-ins can be used as a proxy for visitor activity or customer engagement.

Possible check-in variables include:

- total number of check-ins per restaurant
- number of check-ins before the review date
- check-ins per month
- weekday check-ins
- weekend check-ins
- recent check-in activity

If the model is designed to predict satisfaction at the time of a review, only check-ins before the review date should be used. Otherwise, there is a risk of data leakage.

### 9.5 Tip variables

Tips are short customer comments and can indicate engagement or quick customer feedback.

Possible tip variables include:

- number of tips per restaurant
- average tip length
- number of compliments on tips
- sentiment of tips, if used carefully
- number of tips before review date

Tips can be used as additional signals of customer interaction.

### 9.6 Photo variables

Photos may indicate visual engagement with the restaurant.

Possible photo variables include:

- total number of photos per restaurant
- number of food photos
- number of drink photos
- number of menu photos
- number of inside photos
- number of outside photos
- share of food-related photos
- share of interior/exterior photos

These variables may help describe how customers visually present and interact with the restaurant.

### 9.7 External variables

The assignment explicitly mentions that external factors may affect restaurant performance and customer satisfaction. These factors can be included if they are relevant and can be merged meaningfully with Yelp data.

Possible external variables include:

- weather conditions on the review date
- temperature
- precipitation
- snow
- extreme weather indicators
- socio-economic indicators
- median income
- population density
- unemployment rate
- education level
- neighborhood or ZIP-code characteristics

External data are optional. They should only be used if they can be matched reliably by time and location.

A strong justification is:

**External variables are included only if they are theoretically relevant, explainable for restaurant satisfaction, and mergeable with Yelp observations through date and geographic location.**

If external data cannot be merged reliably, the paper can remain focused on Yelp-internal variables and mention external data as a limitation or future extension.

## 10. Data preparation tasks

The data preparation part is a core deliverable of the term paper. It should be documented carefully and transparently.

### 10.1 Restaurant selection

The first step is to select restaurant businesses from the business table.

Example rule:

```text
Keep businesses where categories contains "Restaurants".
```

The number of businesses and reviews before and after filtering should be reported.

### 10.2 Table combination

The final review-level dataset should combine information from several Yelp tables.

Recommended joins:

| Data source | Join key | Preparation before join |
|---|---|---|
| Review table | `business_id`, `user_id` | Main review-level table |
| Business table | `business_id` | Filter for restaurants and extract attributes |
| User table | `user_id` | Add reviewer characteristics |
| Check-in table | `business_id` | Aggregate check-ins by restaurant |
| Tip table | `business_id` | Aggregate tips by restaurant |
| Photo table | `business_id` | Aggregate photos by restaurant |

Important:

Tables such as tips and photos should usually be aggregated before joining, otherwise one review can be duplicated many times.

### 10.3 Data cleaning

Data cleaning should include:

- removing duplicates
- handling missing values
- converting date variables
- transforming JSON attributes into structured columns
- extracting useful business attributes
- standardizing category variables
- checking impossible or inconsistent values
- deciding how to treat closed restaurants
- removing irrelevant non-restaurant observations

### 10.4 Feature engineering

Feature engineering should create meaningful variables for analysis and modelling.

Examples:

- binary satisfaction target
- review text length
- review word count
- user account age at review date
- user rating tendency, if constructed in a leakage-safe way
- number of check-ins per restaurant
- number of tips per restaurant
- number of photos per restaurant
- city-level average rating, if used descriptively or time-safe
- rating deviation from city average, if used descriptively or time-safe
- weekend indicator
- season indicator
- cuisine category indicators
- restaurant attribute indicators
- weather variables matched by date and location

If possible, time-aware variables should be created using only information available before the review date.

### 10.5 Encoding and transformation

Machine-learning models require structured numerical input.

Preparation should include:

- one-hot encoding categorical variables
- converting binary attributes into 0/1 variables
- scaling numerical variables where required
- handling high-cardinality variables such as city, category, or postal code
- splitting data into training and test sets
- fitting preprocessing only on training data inside pipelines where feasible

## 11. Exploratory data analysis

The prepared dataset should be suitable not only for machine learning, but also for exploratory data analysis.

The EDA should help understand the data and generate initial insights before modelling.

Possible EDA questions include:

- How are restaurant ratings distributed?
- What share of reviews are classified as satisfied?
- Does satisfaction differ by city or state?
- Does satisfaction differ by cuisine category?
- Are restaurants with more check-ins more likely to receive positive reviews?
- Do users with more Yelp experience rate differently?
- Are certain restaurant attributes associated with higher satisfaction?
- Do restaurants with more photos or tips show different satisfaction levels?
- Are there seasonal or weekday patterns in satisfaction?
- Do weather variables differ between satisfied and dissatisfied reviews?

Useful EDA outputs include:

- descriptive statistics
- frequency tables
- correlation analysis
- satisfaction rates by group
- visualizations of rating distributions
- comparisons between satisfied and dissatisfied reviews
- missingness summaries
- skewness summaries

This section should connect the raw data to the later machine-learning analysis.

## 12. Machine-learning task

The machine-learning task is a supervised binary classification problem.

The target variable is:

```text
satisfied yes/no
```

The goal is to train and compare different models in order to identify the best model for predicting restaurant customer satisfaction.

Suitable models include:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. XGBoost or LightGBM, if available
6. Support Vector Machine, if computationally feasible
7. Neural Network, optional and only if justified

The paper should explain why the selected models are appropriate.

A good model comparison should include both simple and more complex models. Logistic regression can serve as an interpretable baseline, while tree-based models can capture nonlinear relationships and interactions.

## 13. Train-test split and validation

The dataset should be split into training and test data.

A simple approach is:

```text
70 percent training data
30 percent test data
```

Another acceptable approach is an 80/20 split, especially for a large dataset.

A more advanced approach is a time-based split:

```text
Train on older reviews
Test on newer reviews
```

The time-based split is stronger because it better simulates real-world prediction, where past data are used to predict future satisfaction.

Cross-validation can be used during model training, especially for model tuning.

The test set should be used only once at the end for final evaluation.

## 14. Model evaluation

The best model should not be selected based only on accuracy.

Recommended metrics include:

| Metric | Purpose |
|---|---|
| Accuracy | Overall share of correct predictions |
| Balanced accuracy | More robust when the target classes are imbalanced |
| Precision | How many predicted satisfied or dissatisfied customers are truly in that class |
| Recall | How many truly satisfied or dissatisfied customers are correctly identified |
| F1-score | Balance between precision and recall |
| ROC-AUC | Overall classification performance across thresholds |
| PR-AUC | Useful when classes are imbalanced |
| Confusion matrix | Shows false positives and false negatives |
| Lift | Shows usefulness for targeting high-risk or high-opportunity cases |

Class imbalance should be checked. If most reviews are positive, accuracy may be misleading because a model could perform well by mostly predicting “satisfied”.

## 15. Model interpretation

The analysis should explain what drives customer satisfaction. It should not only report predictive performance.

Useful interpretation methods include:

- logistic regression coefficients
- decision tree rules
- feature importance
- permutation importance
- SHAP values, if feasible
- partial dependence plots, if feasible

The interpretation should answer questions such as:

- Which restaurant attributes are associated with satisfaction?
- Do check-ins, tips, and photos improve prediction?
- Do user characteristics matter?
- Are experienced Yelp users more or less likely to give positive reviews?
- Does review timing matter?
- Do external variables improve the model?
- Which factors are practically relevant for restaurant owners or investors?

The language should be careful. Since this is observational data, the results show associations and predictive relationships, not necessarily causal effects.

Use phrases such as:

- is associated with
- is predictive of
- is related to
- contributes to the prediction

Avoid unsupported causal claims such as:

- causes
- proves
- leads to

## 16. Managerial interpretation

The final analysis must be translated into practical insights.

For restaurant owners, the results could support decisions about:

- service attributes
- restaurant atmosphere
- opening hours
- customer engagement
- online reputation management
- review response strategies
- investment in amenities
- understanding customer segments

For investors, the results could support decisions about:

- identifying restaurants with strong satisfaction potential
- evaluating restaurant attractiveness
- comparing locations
- assessing customer engagement signals
- identifying risks in restaurants with weak satisfaction indicators

The term paper should connect model findings to managerial meaning.

Example:

Instead of only saying:

```text
Outdoor seating has high feature importance.
```

write:

```text
Outdoor seating is strongly associated with satisfied reviews. This suggests that physical experience attributes may be relevant for restaurant satisfaction. For owners, investments in seating comfort and atmosphere may therefore be worth evaluating.
```

## 17. Main research question and subquestions

A suitable main research question is:

**How accurately can restaurant customer satisfaction be predicted using Yelp business, review, user, engagement, and contextual data?**

Possible subquestions:

1. Which restaurant attributes are associated with customer satisfaction?
2. Do customer engagement indicators such as check-ins, tips, and photos improve prediction?
3. Do user characteristics improve the prediction of satisfaction?
4. Do review timing and structured review characteristics help explain satisfaction?
5. Do external factors such as weather or socio-economic context add predictive value?
6. Which machine-learning model performs best?
7. What practical recommendations can be derived for restaurant owners or investors?

## 18. Expected deliverables

The term paper should deliver:

1. A clear business problem definition
2. A clear research question
3. A justified focus on the restaurant category
4. A documented unit of analysis
5. A constructed analytical dataset
6. Clearly defined KPIs
7. A binary customer satisfaction target variable
8. Documented data cleaning and feature engineering
9. Exploratory data analysis
10. Machine-learning model comparison
11. Model evaluation using appropriate metrics
12. Interpretation of the best model
13. Managerial recommendations
14. Discussion of limitations
15. Conclusion and outlook

## 19. Suggested term paper structure

A strong paper structure would be:

### 1. Introduction

- Business context
- Importance of restaurant customer satisfaction
- Client perspective
- Research objective

### 2. Data description

- Yelp dataset overview
- Relevant tables
- Restaurant filtering logic
- Schema choice
- External data sources, if used

### 3. Dataset construction

- Unit of analysis
- Table joins
- Aggregation logic
- KPI definitions
- Target variable construction
- Feature engineering
- Missing-value handling

### 4. Exploratory data analysis

- Descriptive statistics
- Satisfaction distribution
- KPI overview
- Differences by city, restaurant type, user behavior, engagement variables, and weather/contextual variables

### 5. Methodology

- Classification problem
- Train-test split
- Models used
- Evaluation metrics
- Validation approach

### 6. Results

- Model performance comparison
- Best model selection
- Confusion matrix
- Feature importance
- Robustness checks, if included

### 7. Managerial implications

- Main satisfaction drivers
- Practical recommendations for restaurant owners
- Investment implications

### 8. Limitations

- Observational data
- Potential data leakage
- Missing demographic variables
- Possible selection bias
- Limitations of Yelp users as a sample
- External data limitations
- Feature ceiling, if supported by results

### 9. Conclusion

- Summary of findings
- Final recommendation
- Future research possibilities

## 20. Key methodological risks

### 20.1 Data leakage

Data leakage is one of the most important risks.

Variables such as restaurant average stars, restaurant review count, total check-ins, total tips, or total photos may include information from after the review date.

If the model uses future information to predict past satisfaction, the model performance becomes unrealistic.

Best solution:

```text
Use only information that was available before the review date.
```

Simpler acceptable solution:

```text
Use aggregate variables but explicitly discuss possible data leakage as a limitation.
```

### 20.2 Row multiplication through incorrect joins

Because Yelp tables have different units of observation, careless joins can duplicate rows.

For example, if reviews are directly joined with photos and tips, one review may appear multiple times if the restaurant has multiple photos and multiple tips.

Best solution:

```text
Aggregate tips, photos, and check-ins at the business level before joining them to the review-level dataset.
```

### 20.3 Target leakage through review text

If satisfaction is defined by review stars, review sentiment may strongly reveal the target.

For this repository's current scope, the project intentionally avoids review text as a model input. This should be treated as a strength because it keeps the task focused on structured predictors rather than sentiment-from-text.

### 20.4 Class imbalance

Restaurant reviews may be mostly positive. If so, a model can achieve high accuracy by predicting most customers as satisfied.

Therefore, evaluation should include F1-score, precision, recall, ROC-AUC, PR-AUC, balanced accuracy, and the confusion matrix.

### 20.5 Correlation vs. causality

The model identifies predictive relationships, not causal effects.

For example, if restaurants with outdoor seating receive more satisfied reviews, this does not automatically prove that outdoor seating causes satisfaction.

The paper should avoid causal claims unless a causal research design is used.

## 21. Minimum viable project scope

A realistic and strong minimum version of the project would be:

- use Yelp restaurant reviews
- define satisfaction as `review_stars >= 4`
- create a review-level dataset
- include business attributes, user metadata, review timing, and check-in or engagement aggregates
- include tip and photo aggregates if feasible
- include external weather data if it can be merged reliably
- perform exploratory data analysis
- train logistic regression, random forest, and gradient boosting models at minimum
- compare models using accuracy, balanced accuracy, F1-score, ROC-AUC, and PR-AUC
- interpret the best model using coefficients or feature importance
- provide managerial recommendations

This version is feasible and directly answers the assignment.

## 22. Advanced extension

A more ambitious version could include:

- time-based train-test split
- leakage-free historical feature construction
- more detailed weather matching
- socio-economic data matched by ZIP code or city
- SHAP-based model interpretation
- comparison of models with and without external variables
- robustness check using alternative satisfaction definitions
- stronger documentation of feature ceiling

These extensions can improve the scientific quality of the project but also increase complexity.

## 23. Final definition of the task

The real task is not simply to analyze Yelp data. The task is to build a structured customer intelligence pipeline for restaurants.

In precise terms:

**We must create a restaurant-focused analytical dataset from Yelp data, define relevant KPIs, operationalize customer satisfaction as a binary yes/no target, carefully combine tables with different units of observation, prepare the data for exploratory analysis and machine learning, train and compare classification models, identify the best model for predicting satisfaction, interpret the main satisfaction drivers, and translate the findings into actionable recommendations for restaurant owners or investors.**

The strongest project design is a review-level classification model where each row represents one restaurant review and the target variable indicates whether the customer was satisfied.
