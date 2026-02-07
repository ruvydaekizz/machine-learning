################################################
# Random Forests, GBM, XGBoost, LightGBM, CatBoost
################################################

# Gerekli kütüphanelerin import edilmesi
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# !pip install catboost
# !pip install xgboost
# !pip install lightgbm

# Console çıktısını düzenleme
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Uyarıların kapatılması
warnings.simplefilter(action='ignore', category=Warning)

# Veri setinin okunması ve projeye dahil edilmesi
df = pd.read_csv("datasets/diabetes.csv")
print(df)

# Bağımlı ve bağımsız değişenin ayrılması
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

################################################
# Random Forests
################################################

# Modeli kuralım
rf_model = RandomForestClassifier(random_state=17)

# Mevcut modelin hiperparametrelerini getirelim.
rf_model.get_params()

# CV ile modeli fit edelim
cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()           # Accuracy sonuç
cv_results['test_f1'].mean()                 # F1 sonuç
cv_results['test_roc_auc'].mean()            # ROC AUC sonuç

# CROSS VALIDATION TEST HATASI/BAŞARI SONUÇLARI 1: (Random Forest)
# Accuracy: 0.75
# F1 Score: 0.61
# ROC AUC: 0.82


# Sıradaki adımda ***Hiperparametre Optimizasyonu*** ile başarılarımızı arttırmaya çalışacağız.

# İlgili parametre değerleri
rf_params = {"max_depth": [5, 8, None],
            "max_features": [3, 5, 7, "auto"],
            "min_samples_split": [2, 5, 8, 15, 20],
            "n_estimators": [100, 200, 500]}


# En iyi parametre değerlerini bulacağız GridSearchCV ile
rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_               # En iyi parametre değerleri

# Final model kurma
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)       

# Final modelde CV yapma
cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# CROSS VALIDATION TEST HATASI/BAŞARI SONUÇLARI 2: (Random Forest)
# Accuracy: 0.76       -- artış oldu
# F1 Score: 0.64       -- artış oldu
# ROC AUC: 0.827        -- artış oldu az da olsa


# Feature Importance

# Hatalarımızı en düşük seviyeye getirilmesine hizmet eden en önemli değişkenleri önem sırasına yer vermek.

def plot_importance(model, feature, num=len(X), save=False):                # num=len(X) : Gösterilecek değişken sayısını verir. 
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': feature.columns})   # DataFrame'e çevirdik.
    
    plt.figure(figsize=(10,10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    # büyükten küçüğe azalan şekilde sıralar ve görselleştirir.
    
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    
    if save:                            # save = True ise png olarak kaydet en önemli feature tablosunu
        plt.savefig('importances.png')


plot_importance(rf_final, X)          # num=5 yazarsak en iyi 5 feature'ı getirir.
                                        # save=True dersek png dosyası olarak kaydeder.

# TABLO YORUM
# Bu tabloya göre en önemli değişken Glucose değişkeniymiş. 
# SkinThickness o kadar önemli değil, diğer değikenler önemli.
# Glucose, BMI, Age en önemlilermiş. Dolayısıyla yeni feature türetirken bunları odağa alabiliriz.


# Analyzing Model Complexity with Learning Curves (Bonus)
# Hyperparameter Optimization with GridSearchCV bölümünde en düşük hatayı/ en yüksek başarıyı veren hiperparametreleri belirlemiştik.
# Elbow yönteminde model karmaşıklığı arttıkça Train Hatası azalırdı, Test hatası artardı. Overfitting olurdu. Bunun önüne geçebilmek adına
# Model karmaşıklığının azaltılabileceğini ifade etmiştik. Burada bu konuyu ele alacağız.

# Model karmaşıklığı modelden modele farklılık gösterir.

# Random Forest için max_depth parametresine odaklanacağız.

# validation_curve isimli bir metot kullanacağız. Fonksiyonlaştıralım

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
            label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
            label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)

val_curve_params(rf_final, X, y, "max_depth", range(1, 11), scoring="roc_auc")


################################################
# GBM
################################################

# Modeli kuralım
gbm_model = GradientBoostingClassifier(random_state=17)

# Mevcut modelin hiperparametrelerini getirelim.
gbm_model.get_params()

# CV ile modeli fit edelim
cv_results = cross_validate(gbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# CROSS VALIDATION TEST HATASI/BAŞARI SONUÇLARI 1: (GBM)
# Accuracy: 0.759
# F1 Score: 0.634
# ROC AUC: 0.825

# İlgili parametre değerleri
gbm_params = {"learning_rate": [0.01, 0.1],
            "max_depth": [3, 8, 10],
            "n_estimators": [100, 500, 1000],
            "subsample": [1, 0.5, 0.7]}

# En iyi parametre değerlerini bulacağız GridSearchCV ile
gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# En iyi parametre değerleri
gbm_best_grid.best_params_

# Final model kurma
gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)

# Final modelde CV yapma
cv_results = cross_validate(gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# CROSS VALIDATION TEST HATASI/BAŞARI SONUÇLARI 2: (GBM) - artış oldu
# Accuracy: 0.774
# F1 Score: 0.663
# ROC AUC: 0.8345


################################################
# XGBoost
################################################

# Modeli kuralım
xgboost_model = XGBClassifier(random_state=17, use_label_encoder=False)

# Mevcut modelin hiperparametrelerini getirelim.
xgboost_model.get_params()

# CV ile modeli fit edelim
cv_results = cross_validate(xgboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# CROSS VALIDATION TEST HATASI/BAŞARI SONUÇLARI 1: (XGBoost)
# Accuracy: 0.7409
# F1 Score: 0.6231
# ROC AUC: 0.7991

# İlgili parametre değerleri
xgboost_params = {"learning_rate": [0.1, 0.01],
                "max_depth": [5, 8],
                "n_estimators": [100, 500, 1000],
                "colsample_bytree": [0.7, 1]}

# En iyi parametre değerlerini bulacağız GridSearchCV ile
xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# Final model kurma
xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

# CV ile modeli fit edelim
cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# CROSS VALIDATION TEST HATASI/BAŞARI SONUÇLARI 2: (XGBoost)  --  artış oldu
# Accuracy: 0.7604
# F1 Score: 0.6414
# ROC AUC: 0.8170


################################################
# LightGBM
################################################

# Modeli kuralım
lgbm_model = LGBMClassifier(random_state=17)

# Mevcut modelin hiperparametrelerini getirelim.
lgbm_model.get_params()

# CV ile modeli fit edelim
cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# CROSS VALIDATION TEST HATASI/BAŞARI SONUÇLARI 1: (LightGBM)
# Accuracy: 0.7474
# F1 Score: 0.6241
# ROC AUC: 0.7990

# İlgili parametre değerleri
lgbm_params = {"learning_rate": [0.01, 0.1],
            "n_estimators": [100, 300, 500, 1000],
            "colsample_bytree": [0.5, 0.7, 1]}

# En iyi parametre değerlerini bulacağız GridSearchCV ile
lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# Final model kurma
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

# CV ile modeli fit edelim
cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# CROSS VALIDATION TEST HATASI/BAŞARI SONUÇLARI 2: (LightGBM)  --  artış oldu
# Accuracy: 0.7643
# F1 Score: 0.6372
# ROC AUC: 0.8147


# Hiperparametre yeni değerlerle deneme
lgbm_params = {"learning_rate": [0.01, 0.02, 0.05, 0.1],
            "n_estimators": [200, 300, 350, 400],
            "colsample_bytree": [0.9, 0.8, 1]}

# En iyi parametre değerlerini bulacağız GridSearchCV ile
lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# Final model kurma
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

# CV ile modeli fit edelim
cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# CROSS VALIDATION TEST HATASI/BAŞARI SONUÇLARI 3: (LightGBM)  --  Accuracy aynı, F1 azalırken, ROC-AUC artış oldu
# Accuracy: 0.7643
# F1 Score: 0.6193
# ROC AUC: 0.8227


# Hiperparametre optimizasyonu sadece n_estimators için.
lgbm_model = LGBMClassifier(random_state=17, colsample_bytree=0.9, learning_rate=0.01)

lgbm_params = {"n_estimators": [200, 400, 1000, 5000, 8000, 9000, 10000]}

# En iyi parametre değerlerini bulacağız GridSearchCV ile
lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# Final model kurma
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

# CV ile modeli fit edelim
cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# CROSS VALIDATION TEST HATASI/BAŞARI SONUÇLARI 4: (LightGBM)  --   değişiklik gözlenmedi
# Accuracy: 0.7643
# F1 Score: 0.6193
# ROC AUC: 0.8227



################################################
# CatBoost
################################################

# Modeli kuralım
catboost_model = CatBoostClassifier(random_state=17, verbose=False)

# CV ile modeli fit edelim
cv_results = cross_validate(catboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# İlgili parametre değerleri
catboost_params = {"iterations": [200, 500],
                "learning_rate": [0.01, 0.1],
                "depth": [3, 6]}

# En iyi parametre değerlerini bulacağız GridSearchCV ile
catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# Final model kurma
catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

# CV ile modeli fit edelim
cv_results = cross_validate(catboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


# CROSS VALIDATION TEST HATASI/BAŞARI SONUÇLARI 1: (CatBoost)
# Accuracy: 0.7721
# F1 Score: 0.6322
# ROC AUC: 0.8420

########################
# Feature Importance  - tüm modeller için en önemli featurelar hangisi görelim
########################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                    ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)
plot_importance(gbm_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)


################################
# Hyperparameter Optimization with RandomSearchCV (BONUS)
################################

rf_model = RandomForestClassifier(random_state=17)

rf_random_params = {"max_depth": np.random.randint(5, 50, 10),
                    "max_features": [3, 5, 7, "auto", "sqrt"],
                    "min_samples_split": np.random.randint(2, 50, 20),
                    "n_estimators": [int(x) for x in np.linspace(start=200, stop=1500, num=10)]}

rf_random = RandomizedSearchCV(estimator=rf_model,
                                param_distributions=rf_random_params,
                                n_iter=100,  # denenecek parametre sayısı
                                cv=3,
                                verbose=True,
                                random_state=42,
                                n_jobs=-1)

rf_random.fit(X, y)

rf_random.best_params_

rf_random_final = rf_model.set_params(**rf_random.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_random_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# CROSS VALIDATION TEST HATASI/BAŞARI SONUÇLARI 1.1: (Random Forest) - (RandomizedSearchCV)
# Accuracy: 0.7682      
# F1 Score: 0.6293      
# ROC AUC: 0.8361       


################################
# Analyzing Model Complexity with Learning Curves (BONUS)
################################

# İlgili Hiperparametrelerin değerlendirilmesi açısından görselleştirme

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
            label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
            label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)


rf_val_params = [["max_depth", [5, 8, 15, 20, 30, None]],
                ["max_features", [3, 5, 7, "auto"]],
                ["min_samples_split", [2, 5, 8, 15, 20]],
                ["n_estimators", [10, 50, 100, 200, 500]]]


rf_model = RandomForestClassifier(random_state=17)

for i in range(len(rf_val_params)):
    val_curve_params(rf_model, X, y, rf_val_params[i][0], rf_val_params[i][1])

rf_val_params[0][1]