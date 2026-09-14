Laptop Price Prediction Model

Intern: Nusrat Company: Excellence Code Solution (ECS) Task: Build a laptop price prediction model

Overview

This project predicts the price of a laptop (in Euro) from its specifications — brand, RAM, CPU, GPU, screen, storage and weight.

Since price is a continuous value, this is a regression problem, not classification. The main model is an Artificial Neural Network (ANN) built with TensorFlow/Keras, with Linear Regression used as a baseline for comparison.

Dataset
Source: Laptop Price Dataset (Kaggle)
Size: 1,275 rows × 15 columns
Target: Price (Euro)
Missing values: none
Results
Model	R² Score	MAE (Euro)
ANN (final model)	0.796	224
Linear Regression (baseline)	0.760	259

The ANN explains roughly 80% of the variation in laptop prices, with an average error of 224 euro. It beats the linear baseline, though by a modest margin — indicating that laptop pricing is largely driven by linear relationships, with the ANN adding value on non-linear feature interactions.

Pipeline
Load & explore — check shape, data types and missing values
Encoding — convert categorical text columns (Company, TypeName, OpSys, CPU_Company, GPU_Company) into numbers using LabelEncoder
Feature extraction — ScreenResolution — one text field held four separate facts; split into Touchscreen, IPS, X_res and Y_res
Feature extraction — Memory — standardised TB to GB, handled dual-drive entries, and produced SSD, HDD, Flash_Storage and Hybrid capacity columns
Cardinality handling — dropped Product (618 unique names in 1,275 rows), kept and encoded CPU_Type and GPU_Type
Feature engineering — PPI — combined X_res, Y_res and Inches into pixels-per-inch, removing 0.99 multicollinearity between the two resolution columns
EDA — correlation heatmap and outlier boxplot
Scaling — StandardScaler, essential for ANN training
Train/test split — 80% train (1,020) / 20% test (255)
ANN — 17 → 64 → 32 → 1, ReLU hidden layers, linear output, MSE loss, Adam optimizer, 100 epochs
Evaluation — R² and MAE on the held-out test set, plus a Linear Regression baseline
Key findings

Most influential features (correlation with price): RAM 0.74 · SSD 0.67 · CPU_Type 0.47 · PPI 0.47

Log transform tested and removed. The price distribution was right-skewed, so a log transform was applied as the standard fix. Testing both versions showed it made results clearly worse:

	R²	MAE
With log	0.24	318 euro
Without log	0.80	227 euro

Log compresses the spread of the target, which drags R² down, and small errors in log space explode after np.exp() — one prediction reached 174,846 euro against a maximum real price of 6,099. The transform was therefore dropped. The decision was made by experiment rather than by following theory.

Where the model struggles. The Actual vs Predicted plot shows tight predictions in the 500–2000 euro range, where most training data lies, and underestimation of premium laptops above 3,000 euro, where samples are scarce.

No overfitting. The training curve shows training and validation loss falling together and flattening around epoch 30. The extra 70 epochs added little — a smaller epoch count would give the same result faster.

How to run
bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
python laptop.py

Keep laptop_price.csv in the same folder as laptop.py. Running the script prints all results to the terminal and saves five graphs as PNG files.

Files
File	Description
laptop.py	Full pipeline, commented step by step
laptop_price.csv	Dataset
correlation_heatmap.png	Feature correlations
price_boxplot.png	Outlier check on price
price_distribution.png	Price distribution shape
training_curve.png	Training vs validation loss
actual_vs_predicted.png	Predictions against real prices
Possible improvements
One-hot encoding instead of LabelEncoder for high-cardinality columns, removing the artificial ordering that label encoding introduces
Early stopping to halt training once validation loss stops improving
Dropout layers to reduce the gap between training and validation loss
More samples in the premium price range, where the model currently underestimates