"""
=====================================================
 LAPTOP PRICE PREDICTION MODEL
=====================================================
Intern    : Nusrat
Company   : Excellence Code Solution (ECS)
Dataset   : Laptop Price Dataset (Kaggle)
Goal      : Predict laptop price (Euro) from its specs
            (brand, RAM, CPU, GPU, screen, weight, etc.)
Type      : Regression problem (price is a continuous number)
Model     : Artificial Neural Network (ANN)

Pipeline:
  1. Load & explore data
  2. Encode text columns into numbers
  3. Extract features from ScreenResolution
  4. Extract features from Memory
  5. Handle remaining text columns
  6. EDA - correlation heatmap
  7. Feature engineering - PPI
  8. EDA - outlier check
  9. Scaling + train/test split
 10. Build & train ANN
 11. Evaluate & visualize results
=====================================================
"""

# ---------- IMPORTS ----------
import pandas as pd                              # library to work with tables (DataFrame)
import matplotlib.pyplot as plt                  # base plotting library - creates the figure/window
import seaborn as sns                            # built on matplotlib, statistical plots in one line
import tensorflow as tf                          # deep learning library for the ANN
from sklearn.preprocessing import LabelEncoder   # tool to convert text categories into numbers
tf.random.set_seed(42)                           ## make training reproducible

# ==========================================================
#  STEP 1: LOAD & EXPLORE
#  Check the data before changing anything: how big it is,
#  which columns are text, and whether any cells are empty.
# ==========================================================
df = pd.read_csv("laptop_price.csv", encoding="latin-1")   # read the CSV file into a table named df

print(df.head())          # first 5 rows - see what the data looks like
print(df.shape)           # (rows, columns) -> (1275, 15)
df.info()                 # each column: number or text (object), and non-null count
print(df.isnull().sum())  # count empty cells in each column -> all zero, no missing data


# ==========================================================
#  STEP 2: ENCODING (text -> numbers)
#  ML models can only do maths on numbers, not text.
#  These columns hold only NAMES (no hidden quantity),
#  so each name simply gets its own number: 0, 1, 2...
# ==========================================================
le = LabelEncoder()   # create the encoder object (reused for every column)

# --- Company: laptop brand (Apple, HP, Dell...) ---
print(df["Company"].unique())                     # category names before encoding
df["Company"] = le.fit_transform(df["Company"])   # fit = learn categories, transform = replace with numbers
print(df["Company"].unique())                     # confirm numbers replaced the text

# --- TypeName: laptop type (Notebook, Ultrabook, Gaming...) ---
print(df["TypeName"].unique())
df["TypeName"] = le.fit_transform(df["TypeName"])
print(df["TypeName"].unique())

# --- OpSys: operating system (Windows 10, macOS, Linux, No OS...) ---
print(df["OpSys"].unique())
df["OpSys"] = le.fit_transform(df["OpSys"])
print(df["OpSys"].unique())

# --- CPU_Company: processor brand (Intel, AMD, Samsung) ---
print(df["CPU_Company"].unique())
df["CPU_Company"] = le.fit_transform(df["CPU_Company"])
print(df["CPU_Company"].unique())

# --- GPU_Company: graphics card brand (Intel, AMD, Nvidia, ARM) ---
print(df["GPU_Company"].unique())
df["GPU_Company"] = le.fit_transform(df["GPU_Company"])
print(df["GPU_Company"].unique())

df.info()   # 5 columns encoded, 5 text columns still left


# ==========================================================
#  STEP 3: FEATURE EXTRACTION - ScreenResolution
#  One text field holds FOUR separate facts, e.g.
#  "IPS Panel Full HD / Touchscreen 1920x1080".
#  Encoding would lose all of them, so we split it instead.
# ==========================================================
print(df["ScreenResolution"].unique())   # inspect the formats first

# --- Touchscreen: 1 if the text contains "Touchscreen", else 0 ---
df["Touchscreen"] = df["ScreenResolution"].apply(lambda x: 1 if "Touchscreen" in x else 0)
print(df["Touchscreen"].value_counts())   # 188 laptops have a touchscreen

# --- IPS panel: better display quality, found in costlier laptops ---
df["IPS"] = df["ScreenResolution"].apply(lambda x: 1 if "IPS" in x else 0)
print(df["IPS"].value_counts())           # 357 laptops have an IPS panel

# --- Pixel size is always the LAST word, e.g. "1920x1080" ---
resolution = df["ScreenResolution"].str.split().str[-1]   # split on spaces, keep the last word
print(resolution.head(10))

# --- Split "1920x1080" into two numeric columns ---
# Pixel width and height are real quantities that affect price,
# so they must become numbers, not categories.
df["X_res"] = resolution.str.split("x").str[0].astype(int)   # part before 'x' -> width  (1920)
df["Y_res"] = resolution.str.split("x").str[1].astype(int)   # part after  'x' -> height (1080)
print(df[["X_res", "Y_res"]].head())

# --- Drop the original text column: all useful info has been extracted ---
df.drop("ScreenResolution", axis=1, inplace=True)

df.info()   # ScreenResolution gone, 4 new numeric columns added


# ==========================================================
#  STEP 4: FEATURE EXTRACTION - Memory
#  The text holds TWO real quantities: storage size and type
#  (e.g. "128GB SSD +  1TB HDD"). Both affect price, so the
#  field is broken into numeric columns instead of encoded.
# ==========================================================
print(df["Memory"].unique())   # inspect all formats before deciding what to extract

# --- Step 4a: clean the text so only numbers and type names remain ---
df["Memory"] = df["Memory"].str.replace("GB", "")      # "128GB SSD" -> "128 SSD"
df["Memory"] = df["Memory"].str.replace("TB", "000")   # "1TB HDD"   -> "1000 HDD" (same unit as GB)
print(df["Memory"].unique()[:10])

# --- Step 4b: some laptops have two drives, so split at "+" ---
new = df["Memory"].str.split("+", n=1, expand=True)   # break into 2 columns at the first "+"
df["first"]  = new[0].str.strip()                     # e.g. "128 SSD"   (strip removes extra spaces)
df["second"] = new[1].str.strip()                     # e.g. "1000 HDD"  (NaN when there is no second drive)
df["second"] = df["second"].fillna("0")               # laptops with one drive get "0"
print(df[["first", "second"]].head(10))

# --- Step 4c: separate the type (0/1 flag) from the size (number) ---
for value in ["SSD", "HDD", "Flash Storage", "Hybrid"]:                           # loop over the 4 storage types
    df["first_"  + value] = df["first"].apply(lambda x: 1 if value in x else 0)   # 1 if first drive is this type
    df["second_" + value] = df["second"].apply(lambda x: 1 if value in x else 0)  # 1 if second drive is this type

df["first"]  = df["first"].str.replace(r"\D", "", regex=True).astype(int)    # delete non-digits, keep size as number
df["second"] = df["second"].str.replace(r"\D", "", regex=True).astype(int)   # same for the second drive
print(df[["first", "first_SSD", "first_HDD", "second", "second_HDD"]].head())

# --- Step 4d: combine both drives into one column per storage type ---
# size * flag => the size lands in the right column, and 0 everywhere else
df["SSD"]           = df["first"] * df["first_SSD"]           + df["second"] * df["second_SSD"]
df["HDD"]           = df["first"] * df["first_HDD"]           + df["second"] * df["second_HDD"]
df["Flash_Storage"] = df["first"] * df["first_Flash Storage"] + df["second"] * df["second_Flash Storage"]
df["Hybrid"]        = df["first"] * df["first_Hybrid"]        + df["second"] * df["second_Hybrid"]

# --- Remove the original text column and all temporary helper columns ---
df.drop(columns=["Memory", "first", "second",
                 "first_SSD", "first_HDD", "first_Flash Storage", "first_Hybrid",
                 "second_SSD", "second_HDD", "second_Flash Storage", "second_Hybrid"],
        inplace=True)

print(df[["SSD", "HDD", "Flash_Storage", "Hybrid"]].head())
df.info()   # 21 columns now, only 3 text columns left: Product, CPU_Type, GPU_Type


# ==========================================================
#  STEP 5: HANDLE THE REMAINING TEXT COLUMNS
#  Three text columns are left: Product, CPU_Type, GPU_Type.
#  None of them hides a quantity, so extraction is not needed.
#  The decision here is KEEP-AND-ENCODE vs DROP, and it depends
#  on cardinality - how many unique values a column holds.
#  Too many categories = too few samples each = the model
#  memorises names instead of learning patterns (overfitting).
# ==========================================================

# --- Check cardinality before deciding ---
print(df["Product"].nunique())   # 618 unique model names in only 1275 rows (~2 samples each)

# Dropped: with ~2 samples per name the model would overfit, and
# brand/type information is already captured by Company and TypeName.
df.drop("Product", axis=1, inplace=True)

# --- Same test for the last two text columns ---
print(df["CPU_Type"].nunique())   # 93  categories  -> ~14 samples each
print(df["GPU_Type"].nunique())   # 106 categories  -> ~12 samples each

# Kept and encoded: ~12-14 samples per category is enough to learn from,
# and these columns carry real price signal (e.g. Core i7 vs Celeron).
df["CPU_Type"] = le.fit_transform(df["CPU_Type"])   # convert processor names into numbers
df["GPU_Type"] = le.fit_transform(df["GPU_Type"])   # convert graphics card names into numbers

df.info()   # confirm: no 'object' columns left - dataset is fully numeric


# ==========================================================
#  STEP 6: FEATURE ENGINEERING - PPI (pixels per inch)
#  X_res and Y_res are 0.99 correlated - they carry the same
#  information twice, which confuses the model.
#  Combining them with Inches gives PPI: how densely the pixels
#  are packed, i.e. how sharp the screen actually is.
#  Same pixel count on a smaller screen = sharper = costlier.
#  3 columns (X_res, Y_res, Inches) are replaced by 1 better one.
# ==========================================================
df["PPI"] = ((df["X_res"]**2 + df["Y_res"]**2)**0.5) / df["Inches"]   # diagonal pixels / diagonal inches
print(df["PPI"].head())

df.drop(columns=["X_res", "Y_res", "Inches"], inplace=True)   # all three are now contained in PPI


# ==========================================================
#  STEP 7: EDA - CORRELATION HEATMAP
#  Correlation measures how strongly two columns move together,
#  on a scale from -1 to +1:
#     +1 = feature goes up -> price goes up
#      0 = no relationship at all
#     -1 = feature goes up -> price goes down
#  Two different readings matter here:
#     feature vs PRICE   -> closer to +/-1 is BETTER (useful feature)
#     feature vs FEATURE -> closer to +/-1 is WORSE  (duplicate info)
#  Reminder: only trust these values for truly numeric columns.
#  Label-encoded columns (Company, OpSys, GPU_Type...) hold
#  arbitrary category numbers, so their correlation is not meaningful.
# ==========================================================
print(df.corr()["Price (Euro)"].sort_values(ascending=False))   # correlation of every feature with price, strongest first

plt.figure(figsize=(14, 10))                                           # canvas big enough for 18x18 cells
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")         # draw correlation grid with numbers inside
plt.title("Correlation Heatmap")                                       # heading shown above the plot
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")   # save as a print-quality image for the report
plt.show()                                                             # open the window and display it


# ==========================================================
#  STEP 8: EDA - OUTLIER CHECK
#  Outliers are values far away from the rest of the data,
#  e.g. one laptop at 6000 euro when most are 400-2000.
#  They pull the model towards themselves and hurt accuracy,
#  so they must be found before training.
#  A boxplot shows them as separate dots beyond the whiskers.
# ==========================================================
plt.figure(figsize=(8, 5))                                        # smaller canvas - only one column is plotted
sns.boxplot(x=df["Price (Euro)"])                                 # draw a boxplot of the price column
plt.title("Price Distribution - Outlier Check")                   # heading for the plot
plt.savefig("price_boxplot.png", dpi=300, bbox_inches="tight")    # save as a print-quality image for the report
plt.show()                                                        # open the window

print(df["Price (Euro)"].describe())   # summary stats: count, mean, min, max and the 25/50/75% marks 

#================================================================
# STEP 9 : TARGET DISTRIBUTION CHECK
# The boxplot showed a right-skewed price distribution, so a log
# transform was tested here (see the evaluation section for the
# comparison). It made results worse, so it stays disabled and
# the model is trained on raw euro prices.
#================================================================
import numpy as np
# df["Price (Euro)"] = np.log(df["Price (Euro)"])   # tested and disabled - see evaluation section
sns.histplot(df["Price (Euro)"], kde=True)
plt.title("Price Distribution")
plt.savefig("price_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

#==================================================================
# STEP 10: SEPARATE FEATURE (X) AND TARGET (Y)
# X = everything the model may look at (the specs) 
# Y = the answer it must learn to predict ( price)
# price must be removed from X , otherwise the model would simply 
# copy the answer instead of learning the pattern.
#===================================================================
X = df.drop("Price (Euro)", axis=1)                              # all columns except price -> the questions
y = df["Price (Euro)"]                                           # only the price column -> the answers
print(X.shape)                                                   # should be (1275, 17 ) - all rows, price column removed
print(y.shape)                                                   # should be (1275,) - one value per laptop
print(X.columns)                                                 # confirm price is not in the feature list
print(y.head())

#+=========================================================================
# STEP 11: FEATURE SCALING
# THE feature live on very different ranges:
#     Touchscreen -> 0 or 1
#     RAM          -> 2 TO 64
#     PPI          -> 90 TO 350
#     SSD          -> 0 TO 1000
# An ANN computes weighted sums, so a column with large numbers
# drowns out a column with small ones, no matter how important 
# that small column really is.
# Example : SSD 512 + Touchscreen 1 = 513, and without the touchscreen it is 512
# the model  barely sees a difference , even though a touchscreen adds a few hundred euro
# standardscaler rescales every column to means 0 , spread 1, so all features start on equal footing.
#ESSENTIAL for ANN. Tree models ( Random Forest) don't need it, because they only compare values instead of adding them.
#========================================================================================================================
from sklearn.preprocessing import StandardScaler                  #rescales every feature to 0 , spread 1
scaler = StandardScaler()                                         # create the scaler object
X_scaled = scaler.fit_transform(X)                # learn each column's average & spread, then rescale
print(X_scaled[:3])   # first 3 rows after scaling - values should now be small, around -2 to +2

# ==========================================================
#  STEP 12: TRAIN / TEST SPLIT
#  The data is divided into two parts:
#     80% training - the model learns from these laptops
#     20% testing  - hidden away, used only at the very end
#  Why hide part of the data? If the model is tested on the
#  same laptops it learned from, memorisation would look like
#  learning. The test set acts as an exam with unseen questions,
#  which is the only honest measure of real performance.
# ==========================================================
from sklearn.model_selection import train_test_split                                             # splits data into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) 
print(X_train.shape)      #should be about (1020,17) 
print(X_test.shape)       # should be about (255,17)
#============================================================================================
# STEP 13: BUILD THE ANN 
# An ANN is built from layers of "neurons" . Each neuron takes 
# the inputs, multiplies them by weights, adds them up, and passes the result on. 
# stacking layers lets the network  learn patterns that are not simple straight lines - for example
# how brand, RAM and storage type toghter affect price.
#
# Structure used here:
#        input layer  -> 17 features
#        Hidden layer -> 64 neurons, ReLU activation
#     Hidden layer -> 32 neurons, ReLU activation
#     Output layer -> 1 neuron, NO activation (regression)
#
#  The output layer is what makes this a regression network:
#  one neuron because one number is wanted, and no activation
#  so the value is free to be any price, not squeezed into 0-1.
# ===========================================================================================================
from tensorflow.keras.models import Sequential   # a model where layers are stacked one after anothe
from tensorflow.keras.layers import Dense        # Dense = fully connected layer
model = Sequential()   # create an empty model, layers will be added one by one
model.add(Dense(64, activation="relu", input_shape=(X_train.shape[1],)))   # first hidden layer - 64 neurons
model.add(Dense(32, activation="relu"))                                    # second hidden layer - 32 neurons
model.add(Dense(1))   # output layer - 1 neuron, no activation (regression)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])   # set up how the model learns
model.summary()   # print the network structure: layers, output shapes, and parameter counts

# ==========================================================
#  STEP 14: TRAIN THE ANN
#  Training is a repeated loop:
#     1. the model guesses prices for the training laptops
#     2. the loss (MSE) measures how wrong those guesses are
#     3. the optimizer (Adam) nudges the 3,265 weights to reduce it
#     4. repeat
#  One full pass over the training data is called an EPOCH.
#  A slice of 20% is held back from training as a VALIDATION set,
#  so we can watch whether the model is genuinely learning or
#  just memorising: if training loss keeps falling while
#  validation loss starts rising, that is overfitting.
# ==========================================================
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=1)


# ==========================================================
#  STEP 15: EVALUATE ON THE TEST SET
#  These 255 laptops were held out at the very start and the
#  model has never seen them - not for training, not for
#  validation. This is the only honest measure of performance.
#
#  Metrics used (regression, so no "accuracy"):
#     R2  - how much of the price variation the model explains
#           1.0 = perfect, 0.0 = no better than always guessing
#           the average price
#     MAE - average error, in the same units as the target
#
# ==========================================================
y_pred = model.predict(X_test)   # model guesses the price for the 255 unseen laptops
from sklearn.metrics import r2_score, mean_absolute_error   # regression evaluation metrics
y_pred = y_pred.flatten()   # convert (255,1) shape into (255,) so it matches y_test
print("R2 Score:", r2_score(y_test, y_pred))   # how much of the price variation the model explains

# ==========================================================
# ==========================================================
#  LOG TRANSFORM: TRIED AND REMOVED
#  The boxplot showed a right-skewed price distribution, so
#  log(price) was applied first - the standard textbook fix.
#  It was then tested both ways on the test set:
#
#       with log   ->  R2 = 0.24,  MAE = 318 euro
#       without    ->  R2 = 0.80,  MAE = 227 euro
#
#  The log version performed clearly worse, for two reasons:
#    1. log compresses the spread of the target, which drags
#       R2 down even when individual predictions are decent
#    2. a small error in log space explodes after np.exp() -
#       one prediction came out at 174,846 euro when the most
#       expensive laptop in the data is 6,099
#
#  So the transform was dropped and the model is trained on
#  raw euro prices. Decision made by experiment, not by theory.
#  np.clip is kept as a safety net: it holds any prediction
#  inside the real price range seen in the data.
# ==========================================================
# y_pred_euro = np.exp(y_pred)   # (log version - disabled)
# y_test_euro = np.exp(y_test)   # (log version - disabled)
y_pred_euro = y_pred   # no log transform, predictions are already in euros
y_test_euro = y_test   # no log transform, actual values are already in euros

y_pred_euro = np.clip(y_pred_euro, y_test_euro.min(), y_test_euro.max())   # safety net against extreme predictions

print("R2 Score (euro):", r2_score(y_test_euro, y_pred_euro))
print("MAE (euro):", mean_absolute_error(y_test_euro, y_pred_euro))

# ==========================================================
#  STEP 16: TRAINING CURVE
#  The history object recorded the loss after every epoch, for
#  both the training data and the validation split.
#  Plotting them together shows HOW the model learned:
#     both curves falling       -> learning properly
#     training falls, val rises -> overfitting (memorising)
#     both flat and high        -> underfitting (too simple)
#  This graph justifies the choice of 100 epochs in the report.
# ==========================================================
plt.figure(figsize=(10, 5))   # canvas for the training curve
plt.plot(history.history["loss"], label="Training Loss")   # loss on the data the model learned from
plt.plot(history.history["val_loss"], label="Validation Loss")   # loss on the held-out validation split
plt.title("Training vs Validation Loss")                              # heading for the plot
plt.xlabel("Epoch")                                                   # what the horizontal axis means
plt.ylabel("Loss (MSE)")                                              # what the vertical axis means
plt.legend()                                                          # show which line is which
plt.savefig("training_curve.png", dpi=300, bbox_inches="tight")       # save for the report
plt.show()                                                            # open the window
# ==========================================================
#  STEP 17: ACTUAL vs PREDICTED
#  Each dot is one test laptop: its real price on the x-axis
#  and the model's prediction on the y-axis.
#  A perfect model would put every dot on the diagonal line.
#  How tightly the dots hug that line is a visual version of
#  the R2 score - and it shows WHERE the model struggles,
#  which a single number cannot.
# ==========================================================
plt.figure(figsize=(8, 8))                          # square canvas so the diagonal is at 45 degrees
plt.scatter(y_test, y_pred, alpha=0.5)              # one dot per test laptop: actual vs predicted
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)   # perfect prediction line
plt.xlabel("Actual Price (Euro)")                                        # horizontal axis
plt.ylabel("Predicted Price (Euro)")                                     # vertical axis
plt.title("Actual vs Predicted Prices")                                  # heading
plt.savefig("actual_vs_predicted.png", dpi=300, bbox_inches="tight")     # save for the report
plt.show()                                                               # open the window
# ==========================================================
#  STEP 18: BASELINE COMPARISON - LINEAR REGRESSION
#  A more complex model is only worth using if it beats a
#  simple one. Linear Regression is the standard baseline for
#  regression: it can only fit straight-line relationships.
#  Training it on the same data and comparing R2 shows whether
#  the ANN's extra complexity actually bought anything.
# ==========================================================
from sklearn.linear_model import LinearRegression   # simple baseline model
lr = LinearRegression()      # create the baseline model
lr.fit(X_train, y_train)     # train it on exactly the same data as the ANN
lr_pred = lr.predict(X_test)                                       # baseline predictions on the same test set
print("Linear Regression R2:", r2_score(y_test, lr_pred))          # compare against the ANN's R2
print("Linear Regression MAE:", mean_absolute_error(y_test, lr_pred))