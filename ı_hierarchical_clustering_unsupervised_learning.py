##################################################
# Hierarchical Clustering (Unsupervised Learning)
##################################################

# Amaç: Gözlemleri birbirlerine olan benzerliklerine göre alt kümelere ayırmaktır.
# 2 şekilde gerçekleşir:
# 1) Agglomerative Birleştirici : Yukarı doğru kümeler birleştirilerek oluşturulur.
# 2) Divisive Bölümleyici : Aşağı doğru kümeler ayrıştırılarak oluşturulur.

# Kullanılan dataset: Amerika'daki şuçları eyaletlerle birlikte suçlar ve istatistikleri yer almaktadır.. Bu eyaletlere göre segmentlere ayırma yapılacaktır.


# Gerekli kütüphanelerin import edilmesi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

# Veri setinin okutulması
df = pd.read_csv("datasets/USArrests.csv", index_col=0)
df.head()

# Uzaklık temelli bir yöntem kullanacağız. O sebeple değişkenlerin standartlaştırılması önem arz eder.
sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

# linkage yöntemi birleştirici bir clustering yöntemidir. Öklid uzaklığına göre gözlem birimlerini kümelere ayırıyor.
hc_average = linkage(df, "average") 

plt.figure(figsize=(10, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,                  # dendogram: kümeleme yapısını gösteren şemadır.
            leaf_font_size=10)          # leaf_font_size: gözlem birimlerinin index değerlerini görüyor olacağız.
plt.show()
# bu grafiğe göre burada ne kadar küme barındırmamız gerektiğini karar verebiliriz.


plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
            truncate_mode="lastp",          
            p=10,                    # 10 gözlem birimi değerine göre cluster oluştur diyoruz.
            show_contracted=True,
            leaf_font_size=10)
plt.show()

# Hiyerarşik kümeleme yönteminin avantajı bize gözlem birimlerine genelden bakma şansı tanımasıdır, böylelikle kümeleri kolaylıkla oluşturulabilir.

################################
# Küme Sayısını Belirlemek
################################

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.5, color='r', linestyle='--')      #  0.5 noktasına bir çizgi çeker. Bu çizgiler nereden ve kaç kümeye ayıracağız onu temsil eder.
plt.axhline(y=0.6, color='b', linestyle='--')
plt.show()

################################
# Final Modeli Oluşturmak
################################

from sklearn.cluster import AgglomerativeClustering               # birleştirici clustering metodunu import ediyoruz.

cluster = AgglomerativeClustering(n_clusters=5, linkage="average")       # n_clusters = küme sayısı, linkage = birleştirici yöntemi belirler.

clusters = cluster.fit_predict(df)                # modeli eğit ve tahmin et.

df = pd.read_csv("datasets/USArrests.csv", index_col=0)
df["hi_cluster_no"] = clusters

df["hi_cluster_no"] = df["hi_cluster_no"] + 1                 # 1'den başlamaları için düzelttik.


# k-means yöntemi ile hierarchical_clustering değerlerini karşılaştıralım. 
# bunun için k-means klasöründen o değerleri çağıralım. 2 farklı kümeleme yönteminden clusterları değerlendirelim
from h_k_means_unsupervised_learning import clusters_kmeans

df["kmeans_cluster_no"] = df["kmeans_cluster_no"]  + 1
df["kmeans_cluster_no"] = clusters_kmeans

print(df)