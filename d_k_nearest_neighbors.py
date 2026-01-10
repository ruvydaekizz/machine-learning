#################################
# Diabetes Prediction with KNN
#################################

# İş Problemi:

# Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin edebilecek bir makine öğrenmesi modeli geliştirebilir misiniz?

# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır. 
# ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan Pima Indian kadınları 
# üzerinde yapılan diyabet araştırması için kullanılan verilerdir. 768 gözlem ve 8 sayısal bağımsız değişkenden oluşmaktadır. 
# Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

# Değişkenler
# Pregnancies: Hamilelik sayısı
# Glucose: Glikoz.
# BloodPressure: Kan basıncı.
# SkinThickness: Cilt Kalınlığı
# Insulin: İnsülin.
# BMI: Beden kitle indeksi.
# DiabetesPedigreeFunction: Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon.
# Age: Yaş (yıl)
# Outcome: Kişinin diyabet olup olmadığı bilgisi. Hastalığa sahip (1) ya da değil (0)


# Neler Yapacağız?
# 1. Exploratory Data Analysis
# 2. Data Preprocessing
# 3. Model & Prediction
# 4. Model Evaluation
# 5. Hyperparameter Optimization
# 6. Final Model


# Gerekli kütüphaneleri import edelim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# Aykırı değerler için tanımladığımız bazı fonksiyonlar var.
# Eşik değer hesabı yapalım
def outlier_tresholds(dataframe, col_name, q1=0.05, q3=0.95):  # Eşik değer Boxplot(IQR) yönteminde q1=0.25 ve q3=0.75'tir. Bu problem özelinde farklı q1 ve q3 değerleri kullanılmıştır. 
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    
    interquartile_range = quartile3 - quartile1
    
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    
    return low_limit, up_limit

# Bir değişkende aykırı değer var mı yok mu? bunun için fonksiyon kullanalım
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_tresholds(dataframe, col_name)
    
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Hesaplanan eşik değerlerine göre bir değişkende aykırı değer varsa bu aykırı değeri silmesin 
# Ama hesaplanan eşik değerlerle değiştiren bir fonk yazalım
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_tresholds(dataframe, variable)
    
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Görsellerle ilgili bazı ayarları yapalım

pd.set_option('display.max_columns', None)                   # Bütün sütunları göstersin
pd.set_option('display.float_format', lambda x:'%.3f'%x)     # Virgülden sonra 3 basamak göster
# pd.set_option('display.width', 500)                          # Console'da gösterimi geniş tutsun



######################################################
# 1. Keşifçi Veri Analizi (Exploratory Data Analysis)
######################################################

# Veri setini okutalım
df = pd.read_csv("datasets/diabetes.csv")
print(df)

df.head()      # Veri setinin ilk 5 değerini getirelim.
df.shape       # Kaç gözlem birimi var.  (768, 9)      768 satır, 9 sütundan oluşuyor

# Bağımlı ve bağımsız değişkenleri ayrı ayrı daha sonra birlikte değerlendirelim.

####################################################
# Target'ın Analizi (Bağımlı değişken = Outcome)
####################################################

# Outcome: Kategorik bir değişkendir. 1 ve 0'lardan oluşuyor.
# Outcome sınıf dağılımına bakalım

df["Outcome"].value_counts()       # 0 sınıfı: 500,  1 sınıfı: 268

# Sınıf dağılımlarını görelim
sns.countplot(x = "Outcome", data=df)
plt.show()

# Bütün veri göz önünde bulunurularak 1 ve 0 oranına bakalım
ratio = 100 * df["Outcome"].value_counts() / len(df)
print(ratio)                                          # 0 :  65.104 ,  1 : 34.896


####################################################
# Feature'ların Analizi (Bağımsız Değişkenler)
####################################################

# Sayısal değişkenlerin istatisiksel oranı
df.describe().T

# Sayısal değişkenleri görselleştirmek için: kutu grafik veya histogram kullanılır

# Kan basıncı değişkeni için histogram
df["BloodPressure"].hist(bins=20)
plt.xlabel("BloodPressure")
plt.show()                               # 60-80 BloodPressure değerinde yoğunluk mevcut.


# Bir fonksiyon oluşturalım. Tüm sayısal değişkenleri görselleştirelim. (bağımlı değişken dahil)
def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)                         # block=True : grafikler birbirini ezmesin diye

for col in df.columns:                    # fonksiyonu tüm columnlara uyguladık.
    plot_numerical_col(df, col)

# Bağımlı değişkeni dışarıda bırakarak tekrar görselleştirme yapalım
cols = [col for col in df.columns if "Outcome" not in col]              # cols içerisinde sadece bağımsız değişkenler kaldı

for col in cols:                    # fonksiyonu bağımlı değişken hariç tüm columnlara uyguladık.
    plot_numerical_col(df, col)


####################################################################
# Target vs Features (Bağımlı Değişken ve Bağımsız Değişkenler)
####################################################################

# Target'e göre groupby alalım. Bağımsız değişkenler ne şekilde etki etmiş onu görelim
df.groupby("Outcome").agg({"Pregnancies": "mean"})

# Fonksiyonlaştıralım
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end= "\n\n\n")

# Tüm bağımsız değişkenlere uygulayalım
for col in cols:
    target_summary_with_num(df, "Outcome", col)

######################################################
# 2. Data Preprocessing (Veri Ön İşleme)
######################################################

df.shape
df.head()

# Eksik değer analizi
df.isnull().sum()             # Veride eksik değer bulunmamaktadır.

# İstatistiksel dağılımları inceleyelim
df.describe().T

# En başlarda tanımladığımız check_outlier fonksiyonunu tüm bağımsız değişkenlere uygulayalım.
# Hangi bağımsız değişkende aykırı değer var onu görelim

for col in cols:
    print(col, check_outlier(df, col))        # Sadece Insulin değişkeninde aykırı değer var gibi gözüküyor.


# replace_with_tresholds fonksiyonunu getirerek, Insulin değişkeninde var olan aykırı değerleri,
# Insulin değişkeni için hesaplamış olduğumuz thresholds'lar ile değiştirelim

replace_with_thresholds(df, "Insulin")

for col in cols:
    print(col, check_outlier(df, col))        # Tüm değişkenlerdeki aykırı değerlerin gittiğini görüyoruz.

######################################################
# 3. Model Kurma & Prediction
######################################################

######################################
# Standartlaştırma (Standart Scaler)
######################################

for col in cols:            # Bağımsız değişkenlere uyguluyoruz.
    df[col] = StandardScaler().fit_transform(df[[col]])

df.head()            # standarlaştırma sonrası bir check

#############
# Modeling
#############
# Amacımız: Kişilerin özellikleri verildiğinde diyabet olma, diyabet olmama durumlarını tahmin edeceğiz.

y = df["Outcome"]                       # Bağımlı değişken
X = df.drop(["Outcome"], axis=1)        # Bağımsız değişkenler

knn_model = KNeighborsClassifier().fit(X, y)     # Modeli eğitiyoruz

random_user = X.sample(1, random_state=45)       # Rastgele bir kullanıcı seçerek diyabet olup olmama durumunu tahmin edelim
knn_model.predict(random_user)                   # çıktısı: array([1], dtype=int64)

######################################################
# 4. Model Evaluation (Değerlendirme)
######################################################

y_pred = knn_model.predict(X)                    # Bütün veri için tahmin işlemini gerçekleştirelim
print(classification_report(y, y_pred))          # Classification report ile sonuçları değerlendirelim

y_prob = knn_model.predict_proba(X)[:, 1]        # 1 sınıfına ait olma olasılıkları
roc_auc_score(y, y_prob)                         # ROC-AUC değerini bulalım

# İLK TAHMİN SONUÇLARI
# Accuracy: 0.83
# F1 Score: 0.74
# ROC-AUC: 0.90

# Bu sonuçlar: Modeli kurduğumuz veride test ettiğimiz sonuçlar 
# Aslında yapılması gereken şey: Modelin hiç göremediği verideki performansını değerlendirmektir. 
#                                Yoksa ortaya yanlılık çıkar. Bu da doğru değerlendirmemizi engeller.
# 2 yöntem vardı: Hold-Out veya Cross Validation
# Bu veri özelinde Cross Validation yöntemine göre devam edeceğiz.

###################################
# Cross Validation Yöntemine Göre 
###################################
cv_results = cross_validate(knn_model, X, y, cv=5, scoring= ["accuracy", "f1", "roc_auc"])
print(cv_results)

cv_results["test_accuracy"].mean()        
cv_results["test_f1"].mean()              
cv_results["test_roc_auc"].mean()         

# CV TAHMİN SONUÇLARI
# Accuracy: 0.73
# F1-Score: 0.59
# ROC-AUC: 0.78

# Cross Validation yöntemi, İlk Tahmin sonuçlarına göre daha güvenilir bir yöntemdir.
# Çünkü model eğitildiği veri ile test edilmez. Bu sebeple daha doğru sonuçlar verir.


# PEKİ bu başarı skorları nasıl arttırılabilir?
# a. Örnek boyutu arttırılabilir.
# b. Veri Ön İşleme adımı detaylandırılabilir.
# c. Özellik Mühendisliği(Feature Engineering) ile yeni değişkenler türetilebilir.
# d. İlgili algoritma için optimizasyonlar yapılabilir.


######################################################
# 5. Hyperparameter Optimization
######################################################

# Parametre: Modellerin veri içerisinden öğrendiği ağırlıklardır. Ağırlıklar o parametrelerin tahmincileridir.
# Hiperparametre: Kullanıcı tarafından tanımlanması gereken dışsal ve veri seti içerisinden öğrenilemeyen parametrelerdir.

knn_model = KNeighborsClassifier()           # KNN getirelim
knn_model.get_params()                       # Ön tanımlı parametre değerlerini getirir.( n_neighbors: 5 = komşuluk sayısını verir. )

# Amacımız komşuluk sayısını değiştirerek olması gereken en optimum komşuluk sayısının ne olacağını bulmaktır.
# Bunun için bir parametre listesi oluşturuyoruz.
knn_params = {"n_neighbors": range(2, 50)}          # 2'den 50'ye kadar sayılar oluşturduk. Bu sayıları tek tek deniyor olacağız. 
                                                    # En az hatayı veren sayı komşu sayımız olacak demektir.
                                                    # GridSearchCV metodu ile bulacağız.

# En iyi komşu sayısını buluyoruz
knn_gs_best = GridSearchCV(knn_model, knn_params, cv=5, n_jobs=-1, verbose=1).fit(X, y)

# Buradaki cv=5 hiperparametre seçimi için hatamızı 5 katlı değerlendiriyoruz. Diğer CV yöntemi ile karıştırılmamalıdır. 
# n_jobs=-1    : İşlemciler en yüksek performansta çalışsın demektir.
# verbose=1    : Yapılan denemeler sonucunda rapor ekler.

knn_gs_best.best_params_      # En iyi komşuluk sayısı değeridir.        Çıktısı:  {'n_neighbors': 17}


# Hiperparametre optimizasyonu sonrası en iyi komşuluk değeri ile final model kuralım. Modelin başarısının artmasını bekliyoruz.

######################################################
#  6. Final Model
######################################################

# Final modeli kuralım
knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)
# **knn_gs_best.best_params_ değerinin başında ** kullanarak en iyi komşuluk değerini buraya ata demiş olduk.

# Kurulan modelin test hatası sonuçlarına bakalım
cv_results = cross_validate(knn_final, X, y, cv=5,scoring=["accuracy", "f1", "roc_auc"])

cv_results["test_accuracy"].mean()       
cv_results["test_f1"].mean()             
cv_results["test_roc_auc"].mean()        

# FİNAL MODEL SONUÇLARI
# Accuracy: 0.76
# F1 Score: 0.61
# ROC-AUC: 0.81