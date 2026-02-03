# Machine Learning (Makine Öğrenimi) Çalışmaları

Bu depo (repository), Makine Öğrenimi (Machine Learning) alanındaki öğrenim sürecim boyunca oluşturduğum notları, temel algoritma uygulamalarını ve uçtan uca projeleri içermektedir.

Amacım, teorik bilgileri pratik Python kodlarına dökerek hem kendim için bir arşiv oluşturmak hem de bu alana ilgi duyanlara temiz ve anlaşılır örnekler sunmaktır.

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler

Projelerde ağırlıklı olarak aşağıdaki araçlar kullanılmıştır:

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

## 📂 Repo İçeriği ve Dosya Yapısı

Dosyalar, öğrenme sırasına göre alfabetik olarak numaralandırılmıştır. Her dosya belirli bir algoritmanın temel uygulamasını içerir.

| Dosya / Klasör | Açıklama | Kategori |
| :--- | :--- | :--- |
| **📂 projects/**          | Kapsamlı, uçtan uca veri bilimi projeleri. | *Projects* |
| ↳ `house_price_prediction`              | Ev fiyatlarını tahmin eden regresyon projesi. | *Regression* |
| **📄a_simple_linear_regression.py**            | Tek değişkenli basit doğrusal regresyon uygulaması. | *Regression* |
| **📄b_multiple_linear_regression.py**          | Birden fazla bağımsız değişken içeren regresyon analizi. | *Regression* |
| **📄c_logistic_regression.py**          | Sınıflandırma problemleri için Lojistik Regresyon. | *Classification* |
| **📄d_k_nearest_neighbors.py**          | KNN algoritması ile sınıflandırma ve tahmin. | *Classification* |
| **📄e_cart.py**           | Classification and Regression Trees (Karar Ağaçları) uygulaması. | *Trees* |

## 🚀 Öne Çıkan Projeler
### 🏠 House Price Prediction Project
`projects/house_price_prediction_project` dizini altında bulunan bu proje, veri ön işleme (preprocessing), özellik mühendisliği (feature engineering) ve hiperparametre optimizasyonu (GridSearchCV) adımlarını içeren kapsamlı bir çalışmadır.

- **Hedef:** Verilen özelliklere göre ev fiyatlarını en düşük hata payı (RMSE) ile tahmin etmek.
- **Kullanılan Modeller:** Proje kapsamında hem temel hem de gelişmiş topluluk (ensemble) öğrenme algoritmaları karşılaştırmalı olarak kullanılmıştır:
  - **Doğrusal Modeller:** Linear Regression, Ridge, Lasso, ElasticNet
  - **Doğrusal Olmayan Modeller:** KNN, SVR
  - **Ağaç Tabanlı Modeller:** Decision Tree (CART), Random Forest
  - **Gelişmiş Boosting Modelleri:** Gradient Boosting (GBM), XGBoost, LightGBM, CatBoost


## 💻 Kurulum ve Kullanım

Bu repodaki kodları kendi bilgisayarınızda çalıştırmak için:

1. Repoyu klonlayın:
   ```bash
   git clone https://github.com/ruvydaekizz/machine-learning.git
    ```

2. Proje dizinine gidin:
   ```bash
   cd machine-learning
    ```
3. Gerekli kütüphanelerin yüklü olduğundan emin olun (Örnek):
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
   
## 🤝 İletişim
Herhangi bir sorunuz veya öneriniz olursa benimle iletişime geçmekten çekinmeyin!

Bu proje sürekli geliştirilmeye devam etmektedir. Yeni algoritmalar ve projeler eklendikçe repo güncellenecektir.
