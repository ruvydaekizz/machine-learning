####################################################
# Sales Prediction with Linear Regression
####################################################

# Gerekli kütüphaneleri import etme
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

pd.set_option('display.float_format', lambda x: '%2f' %x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

#######################################################
# Simple Linear Regression with OLS Using Scikit-Learn
#######################################################

# Veri setini yükleme
df = pd.read_csv("datasets/advertising.csv")        # dizinde bulunan veri setini okuma 
print(df)

# Toplam 4 tane değişken var. 3 bağımsız ve 1 bağımlı değişken bulunuyor.

# Kaç gözlem olduğuna bakalım
print("Gözlem Bilgisi: ", df.shape)          # 200 gözlem, 4 değişken bulunuyor.

# Mantığı kolay anlamak adına bu dataframe içerisinden 2 değişkeni seçerek ilerleyeceğiz.

X = df[["TV"]]
y = df[["sales"]]

# Bu iki değişken arasında var olduğunu varsaydığımız doğrusal ilişkiyi önce modelleyeceğiz ve 
# daha sonra bu model denklemini bir grafik yardımıyla değerlendireceğiz

################
# Model
################

# model kurma
reg_model = LinearRegression().fit(X, y)

# y_hat = b + w * x
# x = TV 'dir burada.

# sabit(b, bias'ı) getirelim
reg_model.intercept_[0]     # ilgili sabit değerini getirir.

# TV'nin katsayısı(w) getirelim
reg_model.coef_[0][0]


################
# Tahmin
################

# Soru1: 150 birimlik TV harcaması olsa ne kadar satış olması beklenir?

sales_predict = reg_model.intercept_[0] + reg_model.coef_[0][0] * 150
print(sales_predict)
# Eğer 150 birimlik bir TV harcaması olursa 14.163089614080658 birimlik satış olmasını bekliyoruz.

# Soru2: 500 birimlik TV harcaması olsa ne kadar satış olması beklenir?
sales_predict = reg_model.intercept_[0] + reg_model.coef_[0][0] * 500
print(sales_predict)
# Eğer 500 birimlik bir TV harcaması olursa 30.800913765637567 birimlik satış olmasını bekliyoruz.


# istatistiksel özetine bakalım.
df.describe().T

# describe'a göre TV max değeri: 296.400000, biz 500 değerini girdik yani veride olmayan gözlenmemiş bir değeri de hesaplayıp satışın ne olacağını tahmin edebiliriz.


##############################
# Modelin Görselleştirilmesi
##############################

g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9}, ci=False, color="r") 

# regplot = regresyon grafiği oluşturmak için kullanılır
# X = bağımsız değişken
# y = bağımlı değişken
# scatter_kws = grafikte kullanacak olduğumuz renkler ifade edilmiş. b = scatter plotlar(gerçek değerler), s: 9 boyutunu belirtir.
# ci = güven aralığı False = güven aralığı ekleme bilgisi verilmiş
# color = "r"  = regresyon çizgisinin ne renk olacağı ifade edilmiş

g.set_title(f"Model Denklemi: Sales {round(reg_model.intercept_[0], 2)} + TV * {round(reg_model.coef_[0][0], 2)}")
# round : ile virgülden sonra 2 basamak al denilmiş

g.set_ylabel("Satış Sayısı")     # y eksenine Satış Sayısı yaz.
g.set_xlabel("TV Harcamaları")   # x eksenine TV Harcamaları yaz.
plt.xlim(-10, 310)               # -10'dan 310'a kadar y eksenini görselleştir
plt.ylim(bottom=0)               # 0'dan başla 
plt.show()                       # grafiği görselleştir

# Gerçek değerler maviler
# Tahmin edilen değerler kırmızı çizgidir. (Modeldir, tahmin denklemidir.)
# Aralarında doğrusal bir ilişki varmış gibi gözüküyor.
# Dolayısıyla ben bu modele istediğim yerden bir değer sorup tahmin sonucu elde edebiliyorum.(Buradaki sınırlar dahilinde olmasa dahi)


##############################
# Tahmin Başarısı
##############################

# MSE

# y_pred = tahmin edilen değerler (tahmin edilen bağımlı değişenler)
y_pred = reg_model.predict(X)     # X = bağımsız değişkenler

mse = mean_squared_error(y, y_pred)      # y = gerçek değerler, y_pred = tahmin edilen değerler, verirsek bana ortalama hatayı verecektir.
print(mse)        # çıktısı: 10.512652915656757

# Elimizde bir hata değeri var. Olması gereken şey bunun en düşük değeri alması.
# Ne yapacağız?
# y'nin ortalamasına bakacağız
y.mean()    # çıktısı: sales   14.022500        = yani satışların ortalaması 14 birimmiş

# standart sapmasına bakalım
y.std()     # çıktısı: sales   5.217457         = yani 9 ve 19 arasında değerler değişiyor gibi gözüküyor.

# E bu durumda elde etiğimiz 10 değeri büyük mü küçük mü diye düşünecek olursak sanki biraz büyük gibi görünüyor. Yani ort hata 1, 1.5, 2 gibi olsa daha mantıklı gibi

# Zaten satışlarımızın ortalaması 14 birim tahminde hata ortalama 10 birimle hata yapıyorsam e bu o kadar da küçük değil. Bu problem için oldukça yüksek hatta.


# RMSE
rmse = np.sqrt(mse)
print(rmse)         # çıktısı : 3.2423221486546887


# MAE
mae = mean_absolute_error(y, y_pred)
print(mae)          # çıktısı : 2.549806038927486                # Daha küçük çıktı daha mı iyi? Hayır. Modelde değişiklik sonrası 2 MAE'nin veya 2 RMSE'nin kıyası olmalıdır. 


# R-KARE
r_kare = reg_model.score(X, y)
print(r_kare)       # çıktısı : 0.611875050850071  

# R-KARE: Doğrusal regresyon modellerinde modelin başarısına ilişkin önemli bir metriktir. 
# Veri setindeki bağımsız değişkenlerin bağımlı değişkeni açıklama yüzdesidir. 
# Yani TV değişkeninin sales değişkenindeki değişikliği açıklama yüzdesidir. Yani bu modelde bağımsız değişkenler bağımlı değişkenin %61'ini açıklayabilmektedir.

# NOT: değişken sayısı arttıkça R-KARE şişmeye meyillidir. Burada düzeltilmiş R-KARE değerinin de göz önünde bulundurulması gerekir. 