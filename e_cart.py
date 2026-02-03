################################################
# Decision Tree Classification: CART
################################################

# Karar ağaçları yöntemleridir. (Classification and Regression Tree)

# İş Problemi:
# Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin edebilecek bir makine öğrenmesi modeli geliştirebilir misiniz?

# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır. 
# ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan Pima Indian kadınları 
# üzerinde yapılan diyabet araştırması için kullanılan verilerdir. 768 gözlem ve 8 sayısal bağımsız değişkenden oluşmaktadır. 
# Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

# Değişkenler (Diabetes Dataset)
# Pregnancies: Hamilelik sayısı
# Glucose: Glikoz.
# BloodPressure: Kan basıncı.
# SkinThickness: Cilt Kalınlığı
# Insulin: İnsülin.
# BMI: Beden kitle indeksi.
# DiabetesPedigreeFunction: Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon.
# Age: Yaş (yıl)
# Outcome: Kişinin diyabet olup olmadığı bilgisi. Hastalığa sahip (1) ya da değil (0)

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering 
# 3. Modeling using CART
# 4. Hyperparameter Optimization with GridSearchCV
# 5. Final Model
# 6. Feature Importance
# 7. Analyzing Model Complexity with Learning Curves (Bonus)
# 8. Visualizing the Decision Rules 
# 9. Extracting Decision Rules
# 10. Extracting Python/SQL/Excel Codes of Decision Rules
# 11. Prediction using Python Codes
# 12. Saving and Loading Model

# Yüklenmesi gerekenler

# pip install pydotplus
# pip install skompiler
# pip install astor
# pip install joblib

# Gerekli kütüphanelerin yüklenmesi
import warnings            # Uyarılar için gerekli bir kütüphane
import joblib              # Model kaydetme/yükleme için kullanılan kütüphane
import pydotplus           # Graphviz DOT dilini kullanarak grafik ve karar ağaçlarını görselleştiren kütüphane.
import numpy as np
import pandas as pd
import seaborn as sns 
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
# expert_graphviz: Ağacı görselleştirmek amacıyla Graphviz (DOT) formatına dönüştürür.
# export_text: Ağacın karar kurallarını (if-else yapısını) okunabilir metin olarak verir.
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile            # Eğitilmiş makine öğrenmesi modellerini SQL, Excel veya C koduna çevirir.

pd.set_option('display.max_columns', None)
warnings.simplefilter(action='ignore', category=Warning)


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
# 2. Data Preprocessing  (Veri Ön İşleme)
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

###############################
# 3. Modeling using CART
###############################

# Bağımlı ve bağımsız değişkenleri seçme işlemini gerçekleştirelim
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

# Modeli kuralım
cart_model = DecisionTreeClassifier(random_state=1).fit(X, y)      # random_state=1 : aynı rassallıkları alabilmek için tanımlıyoruz.

# Bütün gözlemler için tahmin edilen değişkenleri hesaplayalım
# Confusion matrix için y_pred
y_pred = cart_model.predict(X)

# ROC eğirisi için 1.sınıfa ait olma olasılıklarını hesaplayalım
y_prob = cart_model.predict_proba(X)[:, 1]

# Confusion matrix
print(classification_report(y, y_pred))

# AUC Skor
print(roc_auc_score(y, y_prob))

# İLK BAŞARI SONUÇLARI
# Precision: 1.00
# Recall: 1.00
# F1-Score: 1.00
# Accuracy: 1.00
# ROC-AUC: 1.00

# Başarı nasıl 1 çıkabilir? Overfitting'e mi düştük? Yoksa gerçek sonuçlar bunlar mı?.
# Bu sonuçları/başarıları/hataları nasıl daha doğru değerlendirebiliriz? 
# Hold-Out ve Cross Validation yöntemlerini kullanarak tekrardan değerlendirme yapalım.

###############################################
# HOLD-OUT Yöntemi ile Başarı Değerlendirme
###############################################

# Veri setini train ve test seti olarak 2'ye ayırır. Train seti ile model kurup, Test seti ile test ediyoruz.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

# Train setine yönelik olarak model kuralım.
cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

# Train ve Test hatalarını inceleyelim

# Train Hatası
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)

# HOLD_OUT TRAIN HATASI / BAŞARISI SONUÇLARI: Modeli train seti ile eğitip, train seti ile test ettik burada. 
# Precision: 1.00
# Recall: 1.00
# F1-Score: 1.00
# Accuracy: 1.00
# ROC-AUC: 1.00


# Test Hatası
y_pred = cart_model.predict(X_test)                        # X_test: Modelin hiç görmediği bağımsız değişken değerleridir.
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)

# HOLD_OUT TEST HATASI / BAŞARISI SONUÇLARI: Modeli train seti ile eğitip, modelin hiç görmediği test seti ile test ettik burada. 
# Precision: 0.58
# Recall: 0.57
# F1-Score: 0.58
# Accuracy: 0.71
# ROC-AUC: 0.67

# HOLD-OUT yöntemi sonucunda:
# Model Train setini çok iyi öğrendi ezberledi, 
# Model Test setinde yani hiç görmediği verideki sonuçlarda oldukça kötü sonuçlar verdi. 
# Yani OVERFITTING oldu.


# random_state 'i değiştirip tekrar bakalım.
# Başka bir konu random_state = 17 vermiştik. Bu defa random_state = 45 verelim. Sonuçları tekrardan gözden geçirelim.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=45)

cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

# Train Hatası/ Başarısı
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)


# HOLD_OUT TRAIN HATASI / BAŞARISI SONUÇLARI: Modeli train seti ile eğitip, train seti ile test ettik burada.  (random_state = 45)
# Precision: 1.00
# Recall: 1.00
# F1-Score: 1.00
# Accuracy: 1.00
# ROC-AUC: 1.00 

# Test Hatası/ Başarısı
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)

# HOLD_OUT TEST HATASI / BAŞARISI SONUÇLARI: Modeli train seti ile eğitip, modelin hiç görmediği test seti ile test ettik burada.(random_state = 45)
# Precision: 0.53
# Recall: 0.56
# F1-Score: 0.54
# Accuracy: 0.69
# ROC-AUC: 0.65


# random_state = 17 olunca;
# Train hatası ile Test Hatası arasında olağanüstü bir fark var.
# Train Seti ezberlemiş. Test seti düşük başarı göstermiş
# Sonuçlar yine OVERFITTING olmuş

# Burada CROSS VALIDATION'a gitmemiz gerekiyor.

#####################################################
# CROSS VALIDATION Yöntemi ile Başarı Değerlendirme
#####################################################

# modeli eğitelim

cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)          # Model burada fit yazılsa hadi CV daha sonra bu fit etme işlemini 
                                                                        # görmezden gelecek ve kendisi fit edip, başarı değerlendirme işlemlerini yapacaktır.

cv_results = cross_validate(cart_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# CROSS VALIDATION TEST HATASI/BAŞARI SONUÇLARI: 
# Accuracy: 0.70
# F1-Score: 0.57
# ROC-AUC: 0.67

# Peki bu sonuçlardan en doğrusu hangisidir?
# ***CROSS VALIDATION ile elde edilen sonuçlar en doğru en geçerli başarılardır.***


# Peki sonuçları düşük model başarısını nasıl arttırabiliriz?
# 1_ Örnek boyutu arttırılabilir.
# 2_ Veri Ön işleme adımları detaylandırılabilir.
# 3_ Özellik Mühendisliği (yeni değişkenler türetilebilir.)
# 4_ İlgili Algoritma için optimizasyonlar yapılabilir. (Hiperparametre optimizasyonu)
# 5_ Bu problem özelinde *dengesiz veri yaklaşımları* başarımızı arttırmaya yardımcı olabilir.

# Dengesiz veri yaklaşımı nedir?
# Bağımlı değişkendeki iki farklı sınıfın (0, 1 olsun). Bunları biribirine yaklaştırarak dengesizliği gidermeye çalıştığımız yöntemlerdir.
# - Fazla olan sınıf azaltılabilir.
# - Az olan sınıf arttırılabilir.
# - Rastgele örneklem yöntemi seçilebilir. 

# Sıradaki adımda ***Hiperparametre Optimizasyonu*** ile başarılarımızı arttırmaya çalışacağız.

####################################################
# 4. Hyperparameter Optimization with GridSearchCV
####################################################

# Mevcut modelin hiperparametrelerini getirelim.
cart_model.get_params()

# Burada 2 önemli parametre var. Bunlar Overfitting'in önüne geçecekler. (ön tanımlı değerleri ile gelirler)
# 'min_samples_split' : 2               # Kaç tane kalıncaya kadar bölsün
# 'max_depth' : None                    # Bir ağacın derinliğinin ne kadar olması gerektiği ile ilgili hiperparametredir.


# En az hatayı veren hiperparametre değerlerini GridSearchCV ile seçeceğiz.

cart_params = {'max_depth': range(1, 11),
                'min_samples_split': range(2, 20)}


# GridSearchCV metodunu getirerek bu parametrelere göre arama yapalım.

cart_best_grid = GridSearchCV(cart_model, cart_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# cv = 5 : kaç katlı çapraz doğrulama yapacak 
# n_jobs=-1 : işlemciler tam performans çalışsın
# verbose= True : raporlama yap der. True yerine 1 de yazılabilir. 2 yazarsak : olası tüm kombinasyonları dener ekrana yazdırır

# çıktısı: Fitting 5 folds for each of 180 candidates, totalling 900 fits
# Yani bu 2 parametrenin olası 180 kombinasyonu varmış, toplamda 900 fit işlemi gerçekleşecekmiş

# NOT: Hiperparametre optimizasyonunun sadece Train değil tüm veriye uygulanması daha çok önerilir.
# NOT: Değişkenleri standartlaştırma yapmadık. Ağaç yöntemlerinde standartlaştırmaya ihtiyacımız yoktur.

# En iyi hiperparametre değerlerini getirelim
cart_best_grid.best_params_              # çıktısı: {'max_depth': 5, 'min_samples_split': 4}

# Peki bunlara karşılık en iyi skor hangisidir?
cart_best_grid.best_score_               

# çıktısı: 0.7500806383159324  (Accuracy'dir. Ön tanımlı değeri budur. Biz Accuracy ile devam edeceğiz. İstersen değiştirilir.)
# F1 değeri için
# cart_best_grid = GridSearchCV(cart_model, cart_params, scoring="f1" cv=5, n_jobs=-1, verbose=True).fit(X, y)   
# cart_best_grid.best_params_          çıktısı: {'max_depth': 4, 'min_samples_split': 2}
# cart_best_grid.best_score_           çıktısı: 0.6395752751155839

# ROC-AUC değeri için
# cart_best_grid = GridSearchCV(cart_model, cart_params, scoring:"roc_auc", cv=5, n_jobs=-1, verbose=True).fit(X, y) 
# cart_best_grid.best_params_          çıktısı: {'max_depth': 5, 'min_samples_split': 19}
# cart_best_grid.best_score_           çıktısı: 0.8020768693221523


# GridSearhCV nesnesi yani cart_best_grid    --->>> aslında en iyi modeli saklar. Final modeli burada. 

# random kullanıcı seçelim
random = X.sample(1, random_state=45)
# random değeri tahmin edelim
cart_best_grid.predict(random)       # çıktısı: array([1], dtype=int64)

# Şimdi final model kuralım

#####################
# 5. Final Model
#####################

# Final modeli kuralım
cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state=17).fit(X, y)

# **cart_best_grid.best_params_   : en iyi hiperparametre değerlerini getirir.

# Final modeli kurduktan sonra parametrelere bakalım.
cart_final.get_params()                      # en iyi max_depth ve min_samples_split değerleri gelmiş. Yani en iyi sonuçları vermiş

# Diğer yolu
# cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X, y)
# cart_final.get_params()

# Final modelin Cross Validation hatalarına bakalım

cv_results = cross_validate(cart_final,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# HİPERPARAMETRE OPTİMİZASYONU SONRASI SONUÇLAR
# Accuracy: 0.75
# F1-Score: 0.61
# ROC-AUC: 0.79

# Bu sonuçlar şunu gösteriyor ki: Hiperparametre optimizasyonu sonrası sonuçlarda kayda değer bir ilerleme var. 
# Daha detaylı bir Veri Ön İşleme ve Feature Engineering ile sonuçlar daha da iyileştirilebilir.


##########################
# 6. Feature Importance
##########################

# Hatalarımızı en düşük seviyeye getirilmesine hizmet eden en önemli değişkenleri önem sırasına yer vermek.

cart_final.feature_importances_       # Değişkenlerin önem düzeyini getirir. 

# Ancak bu şekilde ayırt etmek zor olacaktır. 
# Hangi değişken hangi önem düzeyine sahip bunun için bir fonksiyon ele alalım. Görselleştirelim.

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


plot_importance(cart_final, X)          # num=5 yazarsak en iyi 5 feature'ı getirir.
                                        # save=True dersek png dosyası olarak kaydeder.

# TABLO YORUM
# Bu tabloya göre en önemli değişken Glucose değişkeniymiş. 
# SkinThickness o kadar önemli değil, diğer değikenler önemli.
# Glucose, BMI, Age en önemlilermiş. Dolayısıyla yeni feature türetirken bunları odağa alabiliriz.


###############################################################
# 7. Analyzing Model Complexity with Learning Curves (Bonus)
###############################################################

# Hyperparameter Optimization with GridSearchCV bölümünde en düşük hatayı/ en yüksek başarıyı veren hiperparametreleri belirlemiştik.
# Elbow yönteminde model karmaşıklığı arttıkça Train Hatası azalırdı, Test hatası artardı. Overfitting olurdu. Bunun önüne geçebilmek adına
# Model karmaşıklığının azaltılabileceğini ifade etmiştik. Burada bu konuyu ele alacağız.

# Model karmaşıklığı modelden modele farklılık gösterir.

# CART için min_samples_split ve max_depth parametrelerine odaklanacağız.

# validation_curve isimli bir metot kullanacağız.

train_score, test_score = validation_curve(cart_final, X, y,
                                            param_name="max_depth",
                                            param_range=range(1, 11),
                                            scoring="roc_auc",
                                            cv=10)
# param_name="max_depth"    : bu parametreye göre öğrenme eğrilerini yazdırmak istiyoruz
# param_range=(1,11)        : max_depth'in 1'den 11'e kadar derinlik sayılarını denesin.
# roc_auc üzerinden raporlasın
# cv=10   :    10k cv yapsın

print(train_score)
print(test_score)

# train_score ve test_score'un ortalamalarını alalım.
mean_train_score = np.mean(train_score, axis=1)
mean_test_score = np.mean(test_score, axis=1)

# Test(Validation) seti görselleştirelim
plt.plot(range(1, 11), mean_train_score,
        label="Training Score", color='b')

plt.plot(range(1, 11), mean_test_score,
        label="Validation Score", color='g')

plt.title("Validation Curve for CART")
plt.xlabel("Number of max_depth")
plt.ylabel("AUC")
plt.tight_layout()
plt.legend(loc='best')
plt.show()

# **Grafik çıktı yorum:
# max_depth = 2 ve 3 iken Training Score ve Validation Score AUC değerleri birlikte artmış
# max_depth = 4 olduğunda Training Score AUC artıyor. Validation Score AUC azalmaya başlıyor.
# İşte bu noktada model ezberlemeye yani Overfitting olmaya başlıyor.
# Yani model karmaşıklığı/dallanma sayısı arttıkça genellenebilirlik yeteneğini kaybetmeye başlıyor.
# Bu sebeple Training Score ve Validation score yollarını ayıramaya başlamış.


# Hyperparameter Optimization with GridSearchCV bölümünde en uygun değerleri bulmuştuk, max_depth=5 olarak
# Bu grafiğe bakarsak max_depth =3 olarak görünüyor. Peki değiştirmeli miyiz? HAYIR.
# O kısımda diğer hiperparametre değerleriyle birlikte değerlendirme yapmıştık. O sebeple onu baz alacağız.
# Bu bölümde fikir edinmek adına, doğru çıkarımlar yapabiliyor muyuz görmek adına kontrol sağlıyoruz.

# Bu sebeple diğer hiperparametre değerleriyle bir arada değerlendirip bulunan max_depth=5 değeri tutarlı görünüyor.

#######################
# Fonksiyonlaştıralım
#######################

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    
    train_score, test_score = validation_curve(model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)
    
    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)
    
    plt.plot(param_range, mean_train_score, label="Training Score", color="b")
    plt.plot(param_range, mean_test_score, label="Validation Score", color="g")
    
    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show(block=True)

# Fonksiyonlaştırdığımız halini deneyelim - ilgili hiperparametrenin ilgili bir aralığı öğrenme eğrilerini oluşturur
val_curve_params(cart_final, X, y, "max_depth", range(1,11))  # max_depth'e göre üsttekinin aynısı grafik gelir.

# F1 score'a göre bakalım
# val_curve_params(cart_final, X, y, "max_depth", range(1, 11), scoring="f1")


# Peki birden fazla hiperparametre seti olduğunda ne yapacağız? Liste oluşturacağız, sonra o listeyi gezip fonksiyonu uygulayacağız.
cart_val_params = [["max_depth", range(1,11)], 
                    ["min_samples_split", range(2,20)]]

# Şimdi bu listenin elemanlarını gezip val_curve_params fonksiyonunu bunlara uygulamamız lazım.
for i in range(len(cart_val_params)):
    val_curve_params(cart_model, X, y, cart_val_params[i][0], cart_val_params[i][1])

# max_depth ve min_samples_split parametrelerine göre ROC-AUC değerleri ayrı ayrı grafikler halinde gelmiş olur.


#######################################
# 8. Visualizing the Decision Rules 
#######################################

import graphviz    # Graphviz görselleştirme kütüphanesini içeri aktarır
# Model görselleştirmesi için özel bir fonksiyon tanımlıyoruz
def tree_graph(model, col_names, file_name):
    # export_graphviz: Modeli, grafik çizimi için gerekli olan "DOT" veri formatına çevirir.
    # feature_names: Karar kutucuklarında X[0] yerine gerçek değişken isimlerini (Yaş, Gelir vb.) yazar.
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)  # filled=True: Ağaçtaki kutucukları, sınıf ayrımına veya değer yoğunluğuna göre renklendirir.
    
    graph = pydotplus.graph_from_dot_data(tree_str)            # grafiği oluşturma: DOT formatındaki metni alır ve bunu Python'un işleyebileceği bir grafik nesnesine dönüştürür.
    graph.write_png(file_name)                                 # resmi kaydetme: # Oluşturulan bu grafiği PNG formatında bir resim dosyası olarak bilgisayara kaydeder.

# Fonksiyonu Çağırma: 
# 'cart_final' modelini, 'X.columns' içindeki isimleri kullanarak görselleştir ve "cart_final.png" adıyla kaydet.
tree_graph(model=cart_final, col_names=X.columns, file_name="cart_final.png")

# Modelin kurulu olduğu ayarları (hiperparametreleri) listeler (örn: max_depth, min_samples_split).
cart_final.get_params()


#################################
# 9. Extracting Decision Rules
#################################

# Modelin (cart_final) kurallarını, sütun isimlerini (X.columns) kullanarak metne döker ve çıktısını verir.
tree_rules = export_text(cart_final, feature_names=list(X.columns))
print(tree_rules)


###########################################################
# 10. Extracting Python/SQL/Excel Codes of Decision Rules
###########################################################

# Bu işlem Model Deployment (Modeli Canlıya Alma) sürecini çok kolaylaştırır. 
# Modelini bir yazılımcıya, veri analistine veya Excel kullanan bir yöneticiye "al bunu kullan" diye verebilmeni sağlar.

# 1. Modeli saf Python fonksiyonuna çevirir (Scikit-learn bağımlılığı olmadan çalışması için)
print(skompile(cart_final.predict).to('python/code'))

# 2. Modeli SQL sorgusuna çevirir (Veritabanında doğrudan tahmin yapmak için)
print(skompile(cart_final.predict).to('sqlalchemy/sqlite'))

# 3. Modeli Excel formülüne çevirir (Excel'de veriler üzerinde kullanmak için)
print(skompile(cart_final.predict).to('excel'))


#####################################
# 11. Prediction using Python Codes
#####################################

# Bu fonksiyon, Karar Ağacı modelinin öğrendiği kuralları içerir.
# Scikit-learn kütüphanesine ihtiyaç duymadan, sadece standart Python "if-else" mantığıyla çalışır.
# Girdi olarak bir liste (x) alır ve sonuç olarak bir sınıf (0 veya 1) döndürür.

def predict_with_rules(x):
    # x[0], x[1]... gibi ifadeler sırasıyla değişkenlerin değerlerini temsil eder.
    # Bu karmaşık yapı, ağacın dallarındaki tüm "Evet/Hayır" kararlarının iç içe geçmiş halidir.
    return ((((((0 if x[6] <= 0.671999990940094 else 1 if x[6] <= 0.6864999830722809 else
        0) if x[0] <= 7.5 else 1) if x[5] <= 30.949999809265137 else ((1 if x[5
        ] <= 32.45000076293945 else 1 if x[3] <= 10.5 else 0) if x[2] <= 53.0 else
        ((0 if x[1] <= 111.5 else 0 if x[2] <= 72.0 else 1 if x[3] <= 31.0 else
        0) if x[2] <= 82.5 else 1) if x[4] <= 36.5 else 0) if x[6] <=
        0.5005000084638596 else (0 if x[1] <= 88.5 else (((0 if x[0] <= 1.0 else
        1) if x[1] <= 98.5 else 1) if x[6] <= 0.9269999861717224 else 0) if x[1
        ] <= 116.0 else 0 if x[4] <= 166.0 else 1) if x[2] <= 69.0 else ((0 if
        x[2] <= 79.0 else 0 if x[1] <= 104.5 else 1) if x[3] <= 5.5 else 0) if
        x[6] <= 1.098000019788742 else 1) if x[5] <= 45.39999961853027 else 0 if
        x[7] <= 22.5 else 1) if x[7] <= 28.5 else (1 if x[5] <=
        9.649999618530273 else 0) if x[5] <= 26.350000381469727 else (1 if x[1] <=
        28.5 else ((0 if x[0] <= 11.5 else 1 if x[5] <= 31.25 else 0) if x[1] <=
        94.5 else (1 if x[5] <= 36.19999885559082 else 0) if x[1] <= 97.5 else
        0) if x[6] <= 0.7960000038146973 else 0 if x[0] <= 3.0 else (1 if x[6] <=
        0.9614999890327454 else 0) if x[3] <= 20.0 else 1) if x[1] <= 99.5 else
        ((1 if x[5] <= 27.649999618530273 else 0 if x[0] <= 5.5 else (((1 if x[
        0] <= 7.0 else 0) if x[1] <= 103.5 else 0) if x[1] <= 118.5 else 1) if
        x[0] <= 9.0 else 0) if x[6] <= 0.19999999552965164 else ((0 if x[5] <=
        36.14999961853027 else 1) if x[1] <= 113.0 else 1) if x[0] <= 1.5 else
        (1 if x[6] <= 0.3620000034570694 else 1 if x[5] <= 30.050000190734863 else
        0) if x[2] <= 67.0 else (((0 if x[6] <= 0.2524999976158142 else 1) if x
        [1] <= 120.0 else 1 if x[6] <= 0.23899999260902405 else 1 if x[7] <=
        30.5 else 0) if x[2] <= 83.0 else 0) if x[5] <= 34.45000076293945 else
        1 if x[1] <= 101.0 else 0 if x[5] <= 43.10000038146973 else 1) if x[6] <=
        0.5609999895095825 else ((0 if x[7] <= 34.5 else 1 if x[5] <=
        33.14999961853027 else 0) if x[4] <= 120.5 else (1 if x[3] <= 47.5 else
        0) if x[4] <= 225.0 else 0) if x[0] <= 6.5 else 1) if x[1] <= 127.5 else
        (((((1 if x[1] <= 129.5 else ((1 if x[6] <= 0.5444999933242798 else 0) if
        x[2] <= 56.0 else 0) if x[2] <= 71.0 else 1) if x[2] <= 73.0 else 0) if
        x[5] <= 28.149999618530273 else (1 if x[1] <= 135.0 else 0) if x[3] <=
        21.0 else 1) if x[4] <= 132.5 else 0) if x[1] <= 145.5 else 0 if x[7] <=
        25.5 else ((0 if x[1] <= 151.0 else 1) if x[5] <= 27.09999942779541 else
        ((1 if x[0] <= 6.5 else 0) if x[6] <= 0.3974999934434891 else 0) if x[2
        ] <= 82.0 else 0) if x[7] <= 61.0 else 0) if x[5] <= 29.949999809265137
        else ((1 if x[2] <= 61.0 else (((((0 if x[6] <= 0.18299999833106995 else
        1) if x[0] <= 0.5 else 1 if x[5] <= 32.45000076293945 else 0) if x[2] <=
        73.0 else 0) if x[0] <= 4.5 else 1 if x[6] <= 0.6169999837875366 else 0
        ) if x[6] <= 1.1414999961853027 else 1) if x[5] <= 41.79999923706055 else
        1 if x[6] <= 0.37299999594688416 else 1 if x[1] <= 142.5 else 0) if x[7
        ] <= 30.5 else (((1 if x[6] <= 0.13649999350309372 else 0 if x[5] <=
        32.45000076293945 else 1 if x[5] <= 33.05000114440918 else (0 if x[6] <=
        0.25599999725818634 else (0 if x[1] <= 130.5 else 1) if x[0] <= 8.5 else
        0) if x[0] <= 13.5 else 1) if x[2] <= 92.0 else 1) if x[5] <=
        45.54999923706055 else 1) if x[6] <= 0.4294999986886978 else (1 if x[5] <=
        40.05000114440918 else 0 if x[5] <= 40.89999961853027 else 1) if x[4] <=
        333.5 else 1 if x[2] <= 64.0 else 0) if x[1] <= 157.5 else ((((1 if x[7
        ] <= 25.5 else 0 if x[4] <= 87.5 else 1 if x[5] <= 45.60000038146973 else
        0) if x[7] <= 37.5 else 1 if x[7] <= 56.5 else 0 if x[6] <=
        0.22100000083446503 else 1) if x[6] <= 0.28849999606609344 else 0) if x
        [6] <= 0.3004999905824661 else 1 if x[7] <= 44.0 else (0 if x[7] <=
        51.0 else 1 if x[6] <= 1.1565000414848328 else 0) if x[0] <= 6.5 else 1
        ) if x[4] <= 629.5 else 1 if x[6] <= 0.4124999940395355 else 0)

X.columns   # Değişken isimlerini listeler (x listesindeki sıranın ne anlama geldiğini hatırlamak için).

# ÖRNEK 1: Rastgele değerlerden oluşan bir gözlem birimi (hasta/müşteri) tanımlıyoruz.
# Sırasıyla: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
x = [12, 13, 20, 23, 4, 55, 12, 7]

# Fonksiyonu çağırarak bu kişi için tahmin yapıyoruz. Sonuç 0 veya 1 dönecektir.
predict_with_rules(x)

# ÖRNEK 2: Başka bir gözlem birimi
x = [6, 148, 70, 35, 0, 30, 0.62, 50]

# Bu yeni kişi için tahmin fonksiyonunu tekrar çalıştırıyoruz.
predict_with_rules(x)


###################################
# 12. Saving and Loading Model
###################################

# 1. MODELİ KAYDETME
# Eğitilmiş 'cart_final' modelini "cart_final.pkl" adıyla diske kaydeder.
# Bu dosya, modelin öğrendiği tüm matematiksel kuralları saklar.
joblib.dump(cart_final, "cart_final.pkl")

# 2. MODELİ YÜKLEME
# Diskteki .pkl dosyasını okur ve 'cart_model_from_disc' değişkenine atar.
# Artık orijinal modele (cart_final) ihtiyaç duymadan bu değişkenle tahmin yapabiliriz.
cart_model_from_disc = joblib.load("cart_final.pkl")

# 3. YENİ VERİ TANIMLAMA
# Tahmin yapılacak örnek bir veri seti (örneğin bir hasta değerleri) oluşturulur.
x = [12, 13, 20, 23, 4, 55, 12, 7]

# 4. YÜKLENEN MODEL İLE TAHMİN
# x listesi DataFrame'e çevrilir ve .T (transpose) ile yatay hale getirilir (1 satır, 8 sütun).
# Diskten yüklenen model bu veri için tahmin sonucunu (0 veya 1) döndürür.
cart_model_from_disc.predict(pd.DataFrame(x).T)