##################################
# Multiple Linear Regression 
##################################

# Gerekli kütüphaneleri yükleyelim.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%2f' %x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score


# Veri setini okutalım
df = pd.read_csv("datasets/advertising.csv")  
print(df)

X = df.drop("sales", axis=1)         # Bağımsız değişkenleri seçelim
y = df[["sales"]]                    # Bağımlı değişkenleri seçelim


##########################################
# Model Kurma (Hold-Out Yönetimine Göre)
##########################################

# Veriyi %80 train, %20 test seti olarak bölelim.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)  

X_train.shape    # (160,3)
y_train.shape    # (160,1)

X_test.shape     # (40,3)
y_test.shape     # (40,1)   


# Modeli eğitiyoruz.
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)       # X_train: bağımsız değişkenler , y_train: bağımlı değişken


# reg_model = LinearRegression().fit(X_train, y_train)     ----> Bu şekildede yazılabilir

# Sabiti getirelim (b - bias)
reg_model.intercept_         # çıktısı: array([2.90794702])

# Coefficients (w - weights)
reg_model.coef_              # çıktısı: array([[0.0468431 , 0.17854434, 0.00258619]]) - bunlar her bir bağımsız değişkenin ağırlıklarıdır.


#############
# Tahmin 
#############

# Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir? 
# # TV: 30 , radio: 10 , newspaper: 40

# Model denklemini yazınız ve tahmin edilen sales'in ne olduğunu tahmin ediniz.

# sales_predict = b + w1*x1 + w2*x2 + w3*x3
sales_predict = 2.90794702 + 0.0468431*30 + 0.17854434*10 + 0.00258619*40
print(sales_predict)         # çıktısı:  6.20213102


# Bunu fonksiyonel şekilde yapmak istersek? Diyelim ki elimizde yeni veriler olsun ve bunların satış değerini tahmin etmek istiyoruz.
yeni_veri = [[30], [10], [40]]

yeni_veri = pd.DataFrame(yeni_veri).T       # Öncelikle bu verileri DataFrame'e çevirip Transpozunu alıyoruz.
print(yeni_veri)

sales_tahmin = reg_model.predict(yeni_veri)      # predict : tahmin etmek demektir.
print(sales_tahmin)          # çıktısı:  6.202131


#################################
# Tahmin Başarısı Değerlendirme
#################################

# Train seti RMSE hatası
y_pred_train = reg_model.predict(X_train)        # Train setinin tahmin sonucunu bulalım

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))     # train rmse değerini bulalım.
print(train_rmse)                         # çıktısı: 1.736902590147092


# Train R_KARE değerlendirelim 
train_r_kare = reg_model.score(X_train, y_train)     # r_kare: bağımsız değişkenlerin bağımlı değişkeni etkileme, açıklama oranıdır.
print(train_r_kare)                       # çıktısı: 0.8959372632325174


# Test seti RMSE hatasını bulalım. İlk defa train üzerinden kurduğumuz modele test setini soruyoruz.

y_pred_test = reg_model.predict(X_test)         # Test setinin bağımsız değişkenlerini soruyoruz. 

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))  # y_test: bağımlı değişkenin gerçek değerleri,  y_pred_test: bağımlı değişkenin tahmin edilen değerleri
print(test_rmse)                         # çıktısı: 1.4113417558581587


# Test R_KARE değerlendirelim 
test_r_kare = reg_model.score(X_test, y_test)   # Bağımsız değişkenlerin bağımlı değişkeni açıklama yüzdesi %90 civarı. Oldukça yüksek bir açıklama değeridir.
print(test_r_kare)



##################################################
# Model Kurma (Cross Validation Yönetimine Göre)
##################################################

# Veri seti az olduğu için direkt tüm dataset üzerinde CV uyguluyoruz.

# cv = 10
cv_rmse_result = np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error")))
print(cv_rmse_result)                 # çıktısı: 1.6913531708051797

# neg_mean_squared_error: negatif ortalama hatayı verir. Bu sebeple - ile çarptık.
# - 'li kısıma kadar olan parantez içi kısım çıktısı : 10 katlı cv (9'u ile model kur 1'i ile test eder.) MSE hata değerleridir bu çıktılar.
# np.sqrt : RMSE hata değerini verir.
# np.mean : Tüm RMSe hata değerlerinin ortalamasını vermektedir.     çıktısı: 1.6913531708051797


# cv = 5 deneyelim. Veri setimiz az olduğu için
cv_rmse_result = -np.mean(cross_val_score(reg_model, X, y, cv=5, scoring="neg_root_mean_squared_error"))
print(cv_rmse_result)             # çıktısı: 1.7175247278732084

##############################################################
# Peki bu hata sonuçlarına göre hangisine güvenmemiz lazım?
##############################################################
# test_rmse : 1.4113417558581587
# train_rmse : 1.736902590147092
# cv_rmse_result : cv(10) = 1.6913531708051797 , cv(5) : 1.7175247278732084

# Veri setimiz bol olsaydı farketmeyebilirdi. Veri setimiz az olduğu için Cross Validation yöntemine daha fazla güvenmek daha doğru olabilir.