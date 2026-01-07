################################################################
# House Price Predict Model
################################################################

# Kaggle Competition: # https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/evaluation

# Gerekli kütüphanelerin import edilmesi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV


# Uyarıları kapatma - close the warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

# set options
pd.set_option('display.max_columns', None)             # about console results
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)   


#######################################################
# MISSION 1: Apply EDA operations to the dataset
#######################################################

# 1. Genel Resim
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
# 5. Korelasyon Analizi (Analysis of Correlation)

########################################################################################
# Adım 1: Train ve Test veri setlerini okutup birleştiriniz. Birleştirdiğiniz veri üzerinden ilerleyiniz.
########################################################################################

# Read Train and Test datasets
train = pd.read_csv("projects\datasets\house_price_train.csv")
test = pd.read_csv("projects\datasets\house_price_test.csv")

# merge datasets
df = pd.concat([train, test], ignore_index=False).reset_index()                  # concat datasets
df.head()                                                                        # shows first 5 rows

df = df.drop("index", axis=1)                                                    # dropped index column
df.head()                                                                        # again shows first 5 rows


#############################
# 1. Genel Resim
#############################

# quick information about data
def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)                                               # shape information
    print("##################### Types #####################")
    print(dataframe.dtypes)                                              # data types 
    print("##################### Head #####################")
    print(dataframe.head(3))                                             # first 3 rows
    print("##################### Tail #####################")
    print(dataframe.tail(3))                                             # last 3 rows
    print("##################### NA #####################")
    print(dataframe.isnull().sum())                                      # missing values
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1], numeric_only=True).T)      # quatile values


check_df(df)



##################################################
# Numeric ve Kategorik Değişkenlerin Yakalanması (Capturing Numerical and Categorical Variables)
##################################################

def grab_col_names(dataframe, cat_th=10, car_th=20):    # cat_th = 10 ise o kategorik değişken olsun.    car_th = 20 den fazla ise kategorik değil de sayısallık vardır. Onlar artık sınıfsal değilde farklı şekilde yorumlanması gerekmektedir. 
    """
    grab_col_names for given dataframe

    :param dataframe:
    :param cat_th:
    :param car_th:
    :return:
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]     # object tipinde olanları seçiyoruz cat_cols olarak

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                    dataframe[col].dtypes != "O"]                                             # object tipinde olanlar + cat_th < 10 olanları alıyoruz.

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                    dataframe[col].dtypes == "O"]                                             # obejct tipinde olanlar + car_th > 20 olanları alıyoruz

    cat_cols = cat_cols + num_but_cat                                   # kategorik değişkenler
    cat_cols = [col for col in cat_cols if col not in cat_but_car]      # kategorik değişkenlerden kardinal olanları çıkarıyoruz.

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]           # num_cols'da dtype'ı object olmayanlar yanı numeric olanları alıyoruz
    num_cols = [col for col in num_cols if col not in num_but_cat]                          # num_cols'da num_but_cat olanlar gelmesin sadece numeric olanlar gelsin diyoruz.

    print(f"Observations: {dataframe.shape[0]}")           # gözlem sayısı
    print(f"Variables: {dataframe.shape[1]}")              # değişkenler
    print(f'cat_cols: {len(cat_cols)}')                    # kategorik değişken sayıları
    print(f'num_cols: {len(num_cols)}')                    # numerik değişken sayıları
    print(f'cat_but_car: {len(cat_but_car)}')              # kategorik ama kardinal olanların sayıları
    print(f'num_but_cat: {len(num_but_cat)}')              # numerik ama kategorik olanların sayıları
    
    # cat_cols + num_cols + cat_but_car = değişken sayısı.
    # num_but_cat cat_cols'un içerisinde zaten.
    # dolayısıyla tüm şu 3 liste ile tüm değişkenler seçilmiş olacaktır: cat_cols + num_cols + cat_but_car
    # num_but_cat sadece raporlama için verilmiştir.
    
    return cat_cols, cat_but_car, num_cols

cat_cols, cat_but_car, num_cols = grab_col_names(df)           # atama yapılıyor


#####################################################################
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
#####################################################################

# categorik değişkenlerin sayılarını ve oranlarını gösteren fonksiyon
def cat_summary(dataframe, col_name, plot=False):                 # plot=True deseydik grafiklerini de çizerdi, yorumlama açısından kolaylık sağlardı
    print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:             # tüm kategorik değişkenlere uyguluyoruz
    cat_summary(df, col)         # True'da eklersek bu kısıma grafikler açık halde gelecektir.


######################################
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
######################################

# sayısal değişkenlerin verilen quantile bilgilerine göre istatistiksel dağılımlarına ve Histogram dağılımlarına bakalım

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)             # Histogram grafik
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

    print("#####################################")

for col in num_cols:                   # tüm numeric değişkenlere uyguluyoruz
    num_summary(df, col, False)


###############################################################
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
###############################################################

# hedef değişkenin kategorik değişkenlerle analizi

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

for col in cat_cols:                             # hedef değişkenin ortalamasının kategorik değişkenler açısından analizine bakalım
    target_summary_with_cat(df, "SalePrice", col)


# Bağımlı değişkenin incelenmesi
df["SalePrice"].hist(bins=100)
plt.show(block=True)

# Bağımlı değişkenin logaritmasının incelenmesi
np.log1p(df["SalePrice"]).hist(bins=50)
plt.show()


#####################################################
# 5. Korelasyon Analizi (Analysis of Correlation)
#####################################################

corr = df[num_cols].corr()
corr

# Korelasyonların gösterilmesi
sns.set(rc={'figure.figsize': (12,12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

# belirli bir threshold değerine göre correlation matrix oluşturalım
def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    corr = dataframe.corr(numeric_only=True)
    
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    
    if plot:
        sns.set(rc={'figure.figsize':(15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
        
    return drop_list

drop_list = high_correlated_cols(df, plot=True)          # yüksek korelasyona sahip olanlar belirleniyor.
print(drop_list)


######################################
# Mission 2 : Feature Engineering
######################################

###########################################
# Aykırı Değer Analizi (Outlier Analysis)
###########################################

# Aykırı değerlerin baskılanması için up ve lower limit belirleyelim
def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range                   # üst limit belirleme
    low_limit = quantile_one - 1.5 * interquantile_range                    # alt limit belirleme
    return low_limit, up_limit


# Aykırı değer kontrolü
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True                      # aykırı değer var
    else:
        return False                     # aykırı değer yok

for col in num_cols:                      # numeric columnların hangisinde aykırı değer var bakalım
    if col != "SalePrice":                # tahmin edilen değişken hariç hepsini kontroş edelim
        print(col, check_outlier(df, col))


# Aykırı değerlerin baskılanması
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    if col != "SalePrice":                    # tahmin edilen değişken hariç hepsini baskılayalım
        replace_with_thresholds(df,col)



#################################################
# Eksik Değer Analizi (Missing Value Analysis)
#################################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]      # eksik değer olan columnlar

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)                # sıralıyoruz

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)     # oranlarına bakılıyor

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])       # birleştiriliyor

    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

df["Alley"].value_counts()
df["BsmtQual"].value_counts()

# Bazı değişkenlerdeki boş değerler evin o özelliğe sahip olmadığını ifade etmektedir
no_cols = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu",
            "GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]

# Kolonlardaki boşlukların "No" ifadesi ile doldurulması
for col in no_cols:
    df[col].fillna("No",inplace=True)

missing_values_table(df)


# Bu fonsksiyon eksik değerlerin median veya mean ile doldurulmasını sağlar
def quick_missing_imp(data, num_method="median", cat_length=20, target="SalePrice"):
    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir

    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    # değişken object ve sınıf sayısı cat_lengthe eşit veya altındaysa boş değerleri mode ile doldur
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

    # num_method mean ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # num_method median ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data


df = quick_missing_imp(df, num_method="median", cat_length=17)



######################################
# Rare analizi yapınız ve rare encoder uygulayınız.
######################################

# Kategorik kolonların dağılımının incelenmesi

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "SalePrice", cat_cols)

# Nadir sınıfların tespit edilmesi
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


rare_encoder(df,0.01)



################################################################################################
# Yeni değişkenler oluşturunuz ve oluşturduğunuz yeni değişkenlerin başına 'NEW' ekleyiniz.
################################################################################################


df["NEW_1st*GrLiv"] = df["1stFlrSF"] * df["GrLivArea"]

df["NEW_Garage*GrLiv"] = (df["GarageArea"] * df["GrLivArea"])

df["TotalQual"] = df[["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtCond", "BsmtFinType1",
                    "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "FireplaceQu", "GarageQual", "GarageCond", "Fence"]].sum(axis = 1) # 42


# Total Floor
df["NEW_TotalFlrSF"] = df["1stFlrSF"] + df["2ndFlrSF"] # 32

# Total Finished Basement Area
df["NEW_TotalBsmtFin"] = df.BsmtFinSF1 + df.BsmtFinSF2 # 56

# Porch Area
df["NEW_PorchArea"] = df.OpenPorchSF + df.EnclosedPorch + df.ScreenPorch + df["3SsnPorch"] + df.WoodDeckSF # 93

# Total House Area
df["NEW_TotalHouseArea"] = df.NEW_TotalFlrSF + df.TotalBsmtSF # 156

df["NEW_TotalSqFeet"] = df.GrLivArea + df.TotalBsmtSF # 35


# Lot Ratio
df["NEW_LotRatio"] = df.GrLivArea / df.LotArea # 64

df["NEW_RatioArea"] = df.NEW_TotalHouseArea / df.LotArea # 57

df["NEW_GarageLotRatio"] = df.GarageArea / df.LotArea # 69

# MasVnrArea
df["NEW_MasVnrRatio"] = df.MasVnrArea / df.NEW_TotalHouseArea # 36

# Dif Area
df["NEW_DifArea"] = (df.LotArea - df["1stFlrSF"] - df.GarageArea - df.NEW_PorchArea - df.WoodDeckSF) # 73


df["NEW_OverallGrade"] = df["OverallQual"] * df["OverallCond"] # 61


df["NEW_Restoration"] = df.YearRemodAdd - df.YearBuilt # 31

df["NEW_HouseAge"] = df.YrSold - df.YearBuilt # 73

df["NEW_RestorationAge"] = df.YrSold - df.YearRemodAdd # 40

df["NEW_GarageAge"] = df.GarageYrBlt - df.YearBuilt # 17

df["NEW_GarageRestorationAge"] = np.abs(df.GarageYrBlt - df.YearRemodAdd) # 30

df["NEW_GarageSold"] = df.YrSold - df.GarageYrBlt # 48



drop_list = ["Street", "Alley", "LandContour", "Utilities", "LandSlope","Heating", "PoolQC", "MiscFeature","Neighborhood"]

# drop_list'teki değişkenlerin düşürülmesi
df.drop(drop_list, axis=1, inplace=True)



#############################################################
# Label Encoding & One-Hot Encoding işlemlerini uygulayınız.
#############################################################

cat_cols, cat_but_car, num_cols = grab_col_names(df)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and len(df[col].unique()) == 2]    # Object tipte olan ve değeri 2 olan

for col in binary_cols:
    label_encoder(df, col)


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)



##################################
# MODELLEME (Modeling)
##################################

##################################
# Mission 3: Model kurma
##################################

#  Train ve Test verisini ayırınız. (SalePrice değişkeni boş olan değerler test verisidir.)
train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()]

y = train_df['SalePrice'] # np.log1p(df['SalePrice'])
X = train_df.drop(["Id", "SalePrice"], axis=1)

# Train verisi ile model kurup, model başarısını değerlendiriniz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

# modeller
models = [('LR', LinearRegression()),
            #("Ridge", Ridge()),
            #("Lasso", Lasso()),
            #("ElasticNet", ElasticNet()),
            ('KNN', KNeighborsRegressor()),
            ('CART', DecisionTreeRegressor()),
            ('RF', RandomForestRegressor()),
            #('SVR', SVR()),
            ('GBM', GradientBoostingRegressor()),
            ("XGBoost", XGBRegressor(objective='reg:squarederror')),
            ("LightGBM", LGBMRegressor())]
            # ("CatBoost", CatBoostRegressor(verbose=False))]

# modellerin adları ve RMSE değerleri
for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")
    

"""
RMSE: 42558.1491 (LR) 
RMSE: 47557.3947 (KNN) 
RMSE: 38839.5227 (CART) 
RMSE: 28836.8892 (RF) 
RMSE: 25868.1665 (GBM) 
RMSE: 27577.7231 (XGBoost)
RMSE: 28493.8469 (LightGBM)
"""


df['SalePrice'].mean()
df['SalePrice'].std()

##################################################################################
# BONUS : Log dönüşümü yaparak model kurunuz ve rmse sonuçlarını gözlemleyiniz.
# Not: Log'un tersini (inverse) almayı unutmayınız.
##################################################################################

# Log dönüşümünün gerçekleştirilmesi


train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()]

y = np.log1p(train_df['SalePrice'])
X = train_df.drop(["Id", "SalePrice"], axis=1)

# Verinin eğitim ve tet verisi olarak bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)


# lgbm_tuned = LGBMRegressor(**lgbm_gs_best.best_params_).fit(X_train, y_train)

lgbm = LGBMRegressor().fit(X_train, y_train)
y_pred = lgbm.predict(X_test)

y_pred
# Yapılan LOG dönüşümünün tersinin (inverse'nin) alınması
new_y = np.expm1(y_pred)
new_y
new_y_test = np.expm1(y_test)
new_y_test

np.sqrt(mean_squared_error(new_y_test, new_y))
# RMSE : 23732.871096691535


######################################################
# hiperparametre optimizasyonlarını gerçekleştiriniz.
######################################################


lgbm_model = LGBMRegressor(random_state=46)

rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))


lgbm_params = {"learning_rate": [0.01, 0.1],
            "n_estimators": [500, 1500]
            #"colsample_bytree": [0.5, 0.7, 1]
            }

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X_train, y_train)



final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))
print(rmse)           # 0.13392489323314655



################################################################################################################################
# Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.
################################################################################################################################

# feature importance
def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

model = LGBMRegressor()
model.fit(X, y)

plot_importance(model, X)



############################################################################################
# test dataframeindeki boş olan salePrice değişkenlerini tahminleyiniz ve
# Kaggle sayfasına submit etmeye uygun halde bir dataframe oluşturunuz. (Id, SalePrice)
############################################################################################

############################################################################################
# Tahmin ve Submission Oluşturma
############################################################################################

# 1. Modeli Kur (Logaritmik hedef değişken ile)
model = LGBMRegressor()
model.fit(X, y)

# 2. Tahmin Yap (Şu an sonuçlar Logaritmik formatta!)
# "Id" ve "SalePrice" sütunlarını eğitimde kullanmadığımız için çıkarıyoruz
predictions_log = model.predict(test_df.drop(["Id", "SalePrice"], axis=1))

# 3. Logaritmayı Geri Çevir (Inverse Transform)
# Log alınmış fiyatı gerçek fiyata çevirmek için expm1 kullanılır
predictions_real = np.expm1(predictions_log)

# 4. Submission Dosyasını Hazırla
# test_df.index yerine test_df["Id"] kullanıyoruz.
dictionary = {"Id": test_df["Id"].astype(int), "SalePrice": predictions_real}

dfSubmission = pd.DataFrame(dictionary)
dfSubmission.to_csv("housePricePredictions.csv", index=False)

print("Dosya başarıyla oluşturuldu! Yüklenmeye hazır!")

