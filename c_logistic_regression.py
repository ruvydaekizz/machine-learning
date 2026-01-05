######################################################
# Diabetes Prediction with Logistic Regression
######################################################

# İş Problemi:

# Özellikleri belirtildiğinde kişilerin diyabet hastası olup
# olmadıklarını tahmin edebilecek bir makine öğrenmesi
# modeli geliştirebilir misiniz?

# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin
# parçasıdır. ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan
# Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir. 768 gözlem ve 8 sayısal
# bağımsız değişkenden oluşmaktadır. Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun
# pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

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
# 5. Model Validation: Holdout
# 6. Model Validation: 10-Fold Cross Validation
# 7. Prediction for A New Observation


# Gerekli kütüphaneleri import edelim

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns 

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, RocCurveDisplay
from sklearn.model_selection import train_test_split, cross_validate


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
pd.set_option('display.width', 500)                          # Console'da gösterimi geniş tutsun


######################################################
# Keşifçi Veri Analizi (Exploratory Data Analysis)
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
# Data Preprocessing (Veri Ön İşleme)
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
# Standartlaştırma (Robust Scaler)
######################################################

# Neden önemlidir?
# 1) Modellerin değişkenlere eşit yaklaşmasını sağlamamız gerekir.
# 2) Kullanılan parametre tahmin yöntemlerinin daha hızlı ve daha doğru tahminlerde bulunması için standartlaştırma kullanılır.

for col in cols:            # Bağımsız değişkenlere uyguluyoruz.
    df[col] = RobustScaler().fit_transform(df[[col]])

df.head()            # standarlaştırma sonrası bir check


######################################################
# Model Kurma & Prediction
######################################################

# Amacımız: Kişilerin özellikleri verildiğinde diyabet olma, diyabet olmama durumlarını tahmin edeceğiz.

y = df["Outcome"]                       # Bağımlı değişken
X = df.drop(["Outcome"], axis=1)        # Bağımsız değişkenler

log_model = LogisticRegression().fit(X, y)      # Model kurup, eğitelim

sabit = log_model.intercept_            # sabit, b değerini getirelim
print(sabit)

agirlik = log_model.coef_               # ağırlik, w değerini getirelim
print(agirlik)

y_pred = log_model.predict(X)           # tahmin edilen değerleri getirelim

y_pred[0:10]                            # tahmin edilen ilk 10 değer
y[0:10]                                 # gerçek 10 değer



######################################################
# Model Evaluation (Değerlendirme)
######################################################

# Confusion matrix kullanarak değerlendirme yapalım
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title("Accuracy Score: {0}".format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)                 # y=gerçek değerler, y_pred= tahmin edilen değerler


# Classification report üzerinden değerlendirme
print(classification_report(y, y_pred))


# FIRST RESULT
# Accuracy: 0.78
# Presicion: 0.74
# Recall: 0.58
# F1-score: 0.65


# ROC-AUC değerlerine bakalım
y_prob = log_model.predict_proba(X)[:,1]           # y_prob: bağımlı değişkenin 1 sınıfının gerçekleşme olasılığıdır.
roc_auc_score(y, y_prob)                    # ROC-AUC Score: 0.8393955223880598


######################################################
# Model Validation(Doğrulama): Holdout   
######################################################

# veri setini eğitim ve test seti olarak ayıralım.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

# train setine modeli kuralım
log_model = LogisticRegression().fit(X_train, y_train)

# Göstermediğimiz test seti ile test edelim.
y_pred = log_model.predict(X_test)                # y_pred: tahmin edilen değerler

# 1 sınıfına ait olma olasılıklarını bulalım
y_prob = log_model.predict_proba(X_test)[:, 1]


# Başarımızı değerlendirelim
print(classification_report(y_test, y_pred))   # y_test: modelin eğitilirken görmediği set,   y_pred: tahmin edilen değerler

# HOLDOUT RESULTS
# Accuracy: 0.77
# Presicion: 0.79
# Recall: 0.53
# F1-score: 0.63


# ROC-AUC Score
# Eğri verilerini (FPR, TPR) ve AUC Skorunu hesapla
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 6))           # Çerçeve oluşturur.
plt.plot(fpr, tpr, label=f'LogisticRegression (AUC = {auc_score:.2f})') # Modelin mavi çizgisini çiziyoruz
plt.plot([0, 1], [0, 1], 'r--', label='Random Guess') # Kırmızı 'Rastgele Tahmin' çizgisini çiziyoruz

# Başlık ve etiketler
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')           # Legend (Etiketi sağ alta koy)
# Çerçeve ve gösterim
plt.grid(True, alpha=0.3)          # İsteğe bağlı ızgara
plt.show()

# AUC 
roc_auc_score(y_test, y_prob)


#########################################################
# Model Validation(Doğrulama): 10-Fold Cross Validation
#########################################################

y = df["Outcome"]                          # Bağımlı değişken
X = df.drop(["Outcome"], axis=1)           # Bağımsız değişken


log_model = LogisticRegression().fit(X, y)        # Modeli kuruyoruz.

cv_results = cross_validate(log_model, X, y, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])       # CV uygulayalım.

cv_results['test_accuracy'].mean()       # Ortalama accuracy değerimiz        çıktısı: 0.7721925133689839

cv_results['test_precision'].mean()      # Ortalama presicion değerimiz       çıktısı: 0.7192472060223519

cv_results['test_recall'].mean()         # Ortalama recall değerimiz          çıktısı: 0.5747030048916841

cv_results['test_roc_auc'].mean()        # Ortalama ROC-AUC değerimiz         çıktısı: 0.8327295597484277


# CV RESULTS
# Accuracy: 0.77
# Presicion: 0.71
# Recall: 0.57
# F1-score: 0.83


# Buradaki yöntemlere göre aldığımız sonuçlardan hangisine güveneceğiz. CV sonucu en güvenilir yöntemdir.
# Dengesiz bir veri setimiz olduğu için F1-Score ve ROC-AUC değerlerine bakılarak sonuçları değerlendirmek daha doğru olacaktır.

# Amacımız özellikleri verilen kişilerin diyabet olup, olmadığını tahmin etmekti.

# Rastgele seçilen bir kişinin diyabet hastası olup olmama durumunu inceleyelim
X.columns                        # değişkenler
random_user = X.sample(1, random_state=45)        # Rastegele bir kişi seçiyoruz.
print(random_user)

log_model.predict(random_user)        # Rastgele seçilen kişinin diyabet olup, olmama tahmin sonucunu verir.       çıktısı: array([1], dtype=int64)  =  Diyabet hastası