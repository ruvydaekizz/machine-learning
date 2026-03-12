###################################
#  K-Means (Unsupervised Learning)
###################################

# K-Means : Amacımız gözlemleri birbirlerine olan benzerliklerine göre kümelere ayırmaktır.

# Nasıl yapılır?
# Adım 1: Çalışmanın başında küme sayısı belirlenir.
# Adım 2: Rastgele k adet merkez seçilir.
# Adım 3: Her bir gözlem için k merkezlere uzaklıklar hesaplanır.
# Adım 4: Her gözlem en yakın olduğu merkeze yani kümeye atanır.
# Adım 5: Atama işlemlerinden sonra oluşan kümeler için tekrar merkez hesaplamaları yapılır.
# Adım 6: Bu işlem belirlenen bir iterasyon adedince tekrar edilir ve küme içi hata kareler toplamlarının (SSE/ SSR/ SSD) toplamının (total-within-cluster variation) minimum olduğu 
#         durumdaki gözlemlerin kümelenme yapısı nihai kümelenme olarak seçilir.

# Kullanılan dataset: Amerika'daki şuçları eyaletlerle birlikte suçlar ve istatistikleri yer almaktadır.. Bu eyaletlere göre segmentlere ayırma yapılacaktır.


# Gerekli kütüphanelerin import edilmesi
# pip install yellowbrick
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer


# Veri setinin okutulması
df = pd.read_csv("datasets/USArrests.csv", index_col=0) 

df.head()                 # ilk 5 gözlem
df.isnull().sum()         # eksik değer kontrolü
df.info()                 # veri hakkında bilgi edinelim
df.describe().T           # betimsel istatistiklerine bakalım - aykırı değer var mı? anormal bir durum var mı inceliyoruz.


# Uzaklık temelli bir yöntem kullanacağız. O sebeple değişkenlerin standartlaştırılması önem arz eder.
sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)
df[0:5]                      # numpy arrayine dönüştü bu değerler o sebeple bu şekilde gözlemleyebiliriz.


# Modelimizi kuralım
kmeans = KMeans(n_clusters=4, random_state=17).fit(df)     # df = bağımsız değişkenleri (X) ifade eder. Bağımlı değişkenimiz yok çünkü bir gözetimsiz öğrenme yöntemi ile çalışıyoruz.
kmeans.get_params()                                       # hiperparametrelerini inceliyoruz.

kmeans.n_clusters                    # Küme sayısıdır. Dışarıdan ayarlamamız gereken bir değerdir. 
kmeans.cluster_centers_              # Clusterların merkezleridir.
kmeans.labels_                       # Kümelerin etikelerini ifade eder.
kmeans.inertia_                      # SSE/ SSR / SSD değerinin karşılığıdır. Gözlemlerin en yakın cluster'a olan uzaklık değerlerini ifade eder. 



#######################################
# Optimum Küme Sayısının Belirlenmesi 
#######################################

# n_clusters'ın belirlenmesi
# Farklı SSE/SSR/SSD değerlerine göre karar verilmesi gerekmektedir.

kmeans = KMeans()
ssd = []                         # boş bir SSD 
K = range(1, 30)                 

for k in K:                                  # verilen k değerlerinin tümünü girecek.
    kmeans = KMeans(n_clusters=k).fit(df)    # fit etme işlemi gerçekleşecek. 
    ssd.append(kmeans.inertia_)              # inertia_ değerlerini SSD'nin içine gönderecek.


# Görsel oluşturarak yorumlamasını sağlayalım.
plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

# Küme sayısı arttıkça SSE/ SSR/ SSD değerleri düşmüş gibi görünüyor. 
# Gözlem birimi kadar cluster olursa ne olur? 0 olur çünkü her bir gözlem birimi cluster olur, hepsi bir merkez olur. Dolayısıyla SSE 0 olur. 
# Bundan dolayı küme sayısı arttıkça hatanın düşmesini bekleriz.

# Burada direkt algoritmanın bize verdiği küme sayılarına bakılarak direkt iş yapılmaz.
# Unsupervised yöntemlerde direkt buna güvenilememelidir. Bu sadece karar vermede bir yol gösterir. Kendi dokunuşlarına yer vermen gerek.

# Grafiğe göre dirseklenmenin olduğu nokta Optimum nokta diyeceğiz.
# Karar vermek adına eğimin en şiddetli olduğu / dirseklenmenin en şiddetli olduğu nokta seçilir.

# Seçmek adına aşağıdaki yöntemle en uygun optimum küme (n_clusters) belirleyelim. 
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()

elbow.elbow_value_          # en uygun n_clusters değerini vermiş oldu.



####################################
# Final Cluster'ların Oluşturulması
####################################

# Final K-Means modelini kuralım.
kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)          # elbow.elbow_value_ değeri = n_clusters değerini taşımaktadır. 


kmeans.n_clusters                    # Küme sayısıdır. Dışarıdan ayarlamamız gereken bir değerdir. 
kmeans.cluster_centers_              # Clusterların merkezleridir.
kmeans.labels_                       # Kümelerin etikelerini ifade eder.
df[0:5]

# Cluster işlemini yaptık ama hangi eyalet hangi clusterda onu göremiyoruz. 

clusters_kmeans = kmeans.labels_         # kümelerin etiketlerini getirdik. 

df = pd.read_csv("datasets/USArrests.csv", index_col=0)         # veri setini baştan okuttuk.

df["cluster"] = clusters_kmeans         # veri setine yeni bir sütun olarak ekleyelim oluşturduğumuz cluster'ları. Böylelikle hangi eyalet hangi clusterda daha net görebiliriz.

df.head()

df["cluster"] = df["cluster"] + 1        # Cluster sayısı 0 yerine 1 sayısından başlasın diye +1 ekledik. (1,2,3,4,5 olarak Clusterlar ifade edilmektedir.)

# Artık hangi eyalet hangi clusterda bunu biliyoruz.

df[df["cluster"]==5]                    # 5 numaralı clusterda hangi eyaletlerin olduğu bilgisini verir.

# Her bir clusterın kendi içerisindeki gözlem birimlerinin durumlarının ne kadar sağlıklı sonuçlar olup olmadığına bakalım. Göz önünde bulundurulması gereken bir durumdur.
df.groupby("cluster").agg(["count","mean","median"])

df.to_csv("clusters.csv")        # csv dosyasına çevrilip ilgili kişilere çıktı gönderilebilir.