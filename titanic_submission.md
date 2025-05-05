# Laporan Proyek Machine Learning - Roikhatul Janah

## Domain Proyek

Kecelakaan kapal Titanic yang terjadi pada tanggal 15 April 1912 merupakan salah satu tragedi maritim paling terkenal dalam sejarah. Dari 2.224 penumpang dan kru yang ada di kapal, sekitar 1.500 orang meninggal, menjadikannya salah satu bencana maritim damai terbesar dalam sejarah [1]. Tragedi ini memicu peningkatan signifikan dalam regulasi keselamatan maritim dan menjadi titik balik dalam cara industri pelayaran memandang keselamatan penumpang.

Analisis data terkait siapa yang selamat dan tidak selamat dalam tragedi ini memberikan wawasan berharga tentang faktor-faktor yang mempengaruhi tingkat kelangsungan hidup dalam situasi darurat. Analisis prediktif terkait keselamatan penumpang Titanic bukan hanya memiliki nilai historis, tetapi juga aplikasi modern dalam berbagai bidang, termasuk:

1. **Manajemen Keselamatan Transportasi:** Memahami faktor-faktor yang mempengaruhi tingkat keselamatan penumpang dapat membantu dalam pengembangan protokol evakuasi yang lebih efektif dan alokasi sumber daya keselamatan yang lebih baik.

2. **Asuransi dan Manajemen Risiko:** Perusahaan asuransi dapat menggunakan model prediktif semacam ini untuk menilai risiko dan menetapkan premi yang sesuai untuk berbagai jenis perjalanan dan profil demografis.

3. **Pengembangan Kebijakan:** Pembuat kebijakan dapat memanfaatkan wawasan ini untuk mengembangkan regulasi yang lebih efektif terkait keselamatan penumpang pada berbagai moda transportasi.

Proyek ini bertujuan untuk mengembangkan model machine learning yang dapat memprediksi kemungkinan kelangsungan hidup penumpang berdasarkan berbagai karakteristik seperti kelas tiket, jenis kelamin, usia, dan variabel lainnya. Dengan pemahaman yang lebih baik tentang faktor-faktor yang mempengaruhi kelangsungan hidup, kita dapat mengembangkan strategi dan kebijakan yang lebih efektif untuk meningkatkan keselamatan penumpang dalam berbagai situasi darurat.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang yang telah dipaparkan, permasalahan yang akan dibahas dalam proyek ini adalah:

1. Bagaimana kita dapat mengembangkan model prediktif yang akurat untuk memprediksi kelangsungan hidup penumpang berdasarkan karakteristik demografis dan data perjalanan mereka?
2. Fitur-fitur apa yang memiliki pengaruh paling signifikan terhadap kemungkinan kelangsungan hidup penumpang dalam tragedi Titanic?
3. Bagaimana model yang dikembangkan dapat diimplementasikan untuk mendukung pengambilan keputusan terkait keselamatan penumpang di masa mendatang?

### Goals

Tujuan dari proyek ini adalah:

1. Mengembangkan model machine learning dengan akurasi tinggi yang dapat memprediksi kelangsungan hidup penumpang berdasarkan data yang tersedia.
2. Mengidentifikasi dan menganalisis tiga fitur teratas yang memiliki pengaruh paling signifikan terhadap kelangsungan hidup penumpang.
3. Membuat model yang dapat diimplementasikan untuk analisis keselamatan penumpang dan pengambilan keputusan terkait alokasi sumber daya keselamatan.

### Solution Statements

Untuk mencapai tujuan yang telah ditetapkan, beberapa solusi yang diajukan adalah:

1. Mengimplementasikan dan membandingkan beberapa algoritma machine learning untuk klasifikasi, termasuk:
   - Logistic Regression
   - Random Forest Classifier
   - Gradient Boosting Classifier
   - XGBoost Classifier
   
   Pendekatan multi-algoritma ini akan memungkinkan kita untuk mengidentifikasi model dengan performa terbaik berdasarkan metrik evaluasi yang relevan.

2. Melakukan feature engineering dan preprocessing data untuk meningkatkan performa model, termasuk:
   - Penanganan missing values
   - Deteksi dan penanganan outlier
   - Encoding data kategorikal
   - Standarisasi fitur numerik
   - Penerapan teknik oversampling SMOTE untuk mengatasi ketidakseimbangan kelas

3. Melakukan analisis feature importance untuk mengidentifikasi fitur-fitur yang paling berpengaruh terhadap prediksi keselamatan penumpang, yang dapat memberikan wawasan untuk pengambilan keputusan terkait keselamatan transportasi.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah Titanic Dataset yang berisi informasi tentang penumpang kapal Titanic, termasuk apakah mereka selamat atau tidak. Dataset ini dapat diakses secara publik di [Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset).

Dataset ini terdiri dari 891 sampel dengan 12 variabel yang mencakup informasi demografis dan perjalanan para penumpang. Berikut adalah deskripsi variabel-variabel dalam dataset:

### Variabel-variabel pada Titanic Dataset:

1. **PassengerId**: Nomor identifikasi unik untuk setiap penumpang.
2. **Survived**: Variabel target yang menunjukkan apakah penumpang selamat (1) atau tidak (0).
3. **Pclass**: Kelas tiket penumpang (1 = Kelas 1, 2 = Kelas 2, 3 = Kelas 3), yang juga merupakan indikator status sosial-ekonomi.
4. **Name**: Nama penumpang.
5. **Sex**: Jenis kelamin penumpang.
6. **Age**: Usia penumpang dalam tahun.
7. **SibSp**: Jumlah saudara kandung/pasangan yang berada di Titanic.
8. **Parch**: Jumlah orang tua/anak yang berada di Titanic.
9. **Ticket**: Nomor tiket penumpang.
10. **Fare**: Tarif penumpang.
11. **Cabin**: Nomor kabin penumpang.
12. **Embarked**: Pelabuhan embarkasi (C = Cherbourg, Q = Queenstown, S = Southampton).

Berikut adalah statistik deskriptif dari dataset:

```
       PassengerId  Survived  Pclass        Age     SibSp     Parch      Fare
count     891.00    891.00  891.00     714.00    891.00    891.00    891.00
mean      446.00      0.38    2.31      29.70      0.52      0.38     32.20
std       257.35      0.49    0.84      14.53      1.10      0.81     49.69
min         1.00      0.00    1.00       0.42      0.00      0.00      0.00
25%       223.50      0.00    2.00      20.12      0.00      0.00      7.91
50%       446.00      0.00    3.00      28.00      0.00      0.00     14.45
75%       668.50      1.00    3.00      38.00      1.00      0.00     31.00
max       891.00      1.00    3.00      80.00      8.00      6.00    512.33
```

### Eksplorasi Data:

#### Missing Values

Melakukan pengecekan missing values pada dataset:

```
PassengerId    0
Survived       0
Pclass         0
Name           0
Sex            0
Age          177
SibSp          0
Parch          0
Ticket         0
Fare           0
Cabin        687
Embarked       2
```

Dari hasil di atas, terlihat bahwa dataset memiliki missing values pada beberapa kolom:
- **Age**: 177 data hilang (19.9% dari total data)
- **Cabin**: 687 data hilang (77.1% dari total data)
- **Embarked**: 2 data hilang (0.2% dari total data)

#### Distribusi Target

Distribusi target (Survived) pada dataset:
- **Not Survived (0)**: 549 penumpang (61.6%)
- **Survived (1)**: 342 penumpang (38.4%)

Dataset memiliki ketidakseimbangan kelas dengan lebih banyak penumpang yang tidak selamat dibandingkan yang selamat.

#### Korelasi Antar Fitur

Analisis korelasi menunjukkan hubungan berikut dengan variabel target (Survived):
- **Pclass**: Korelasi negatif (-0.34), menunjukkan bahwa penumpang kelas atas memiliki kemungkinan selamat yang lebih tinggi.
- **Sex**: Korelasi kuat terlihat pada visualisasi, dengan penumpang wanita memiliki tingkat keselamatan yang lebih tinggi.
- **Age**: Korelasi negatif lemah (-0.07), dengan penumpang yang lebih muda sedikit lebih mungkin untuk selamat.
- **Fare**: Korelasi positif (0.26), menunjukkan bahwa penumpang dengan tarif lebih tinggi memiliki kemungkinan selamat yang lebih besar.

#### Distribusi Berdasarkan Fitur Kategorikal

Analisis distribusi pada fitur kategorikal menunjukkan:
- **Sex**: Penumpang perempuan memiliki tingkat keselamatan yang jauh lebih tinggi dibandingkan laki-laki.
- **Pclass**: Penumpang kelas 1 memiliki tingkat keselamatan tertinggi, diikuti oleh kelas 2 dan kelas 3.
- **Embarked**: Penumpang yang naik dari pelabuhan Cherbourg memiliki tingkat keselamatan yang sedikit lebih tinggi.

## Data Preparation

Beberapa teknik data preparation yang diterapkan pada dataset adalah sebagai berikut:

### 1. Penanganan Missing Values

Missing values pada dataset ditangani sebagai berikut:
- **Age**: Diisi dengan nilai median (28.0) dari kolom tersebut. Median dipilih karena lebih tahan terhadap outlier dibandingkan mean.
- **Embarked**: Diisi dengan nilai modus (S = Southampton) karena hanya terdapat 2 missing values dan mayoritas penumpang naik dari pelabuhan Southampton.
- **Cabin**: Kolom ini memiliki terlalu banyak missing values (77.1%) sehingga diputuskan untuk dihapus dari analisis.

```python
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
```

### 2. Pemilihan dan Penghapusan Fitur

Beberapa kolom dihapus karena tidak relevan untuk analisis prediktif atau memiliki terlalu banyak missing values:
- **PassengerId**: Hanya identifier dan tidak memiliki nilai prediktif
- **Name**: Terlalu spesifik dan unique untuk digunakan sebagai fitur
- **Ticket**: Format yang tidak konsisten dan nilai prediktif yang rendah
- **Cabin**: Terlalu banyak missing values

```python
data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
```

### 3. Deteksi dan Penanganan Outlier

Untuk fitur numerik ('Age', 'SibSp', 'Parch', 'Fare'), dilakukan deteksi dan penanganan outlier menggunakan metode IQR (Interquartile Range):

```python
def handle_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Capping outlier dengan batas atas dan bawah
    data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])
    data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])
    
    return data
```

Penanganan outlier ini penting karena algoritma machine learning seperti Logistic Regression dapat sensitif terhadap outlier, sehingga dapat mempengaruhi performa model.

### 4. Encoding Data Kategorikal

Fitur kategorikal ('Sex', 'Embarked') diubah menjadi format numerik menggunakan Label Encoding:

```python
label_encoders = {}
categorical_columns = ['Sex', 'Embarked']

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
```

Hasil encoding:
- **Sex**: female = 0, male = 1
- **Embarked**: C = 0, Q = 1, S = 2

Label Encoding dipilih karena fitur-fitur ini memiliki sedikit kategori dan tidak perlu direpresentasikan sebagai one-hot encoding yang dapat meningkatkan dimensionalitas data.

### 5. Standarisasi Fitur Numerik

Untuk memastikan semua fitur berada pada skala yang sama, dilakukan standarisasi fitur menggunakan StandardScaler:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Standarisasi ini penting terutama untuk algoritma yang sensitif terhadap skala fitur seperti Logistic Regression.

### 6. Pembagian Data dan Penanganan Ketidakseimbangan Kelas

Data dibagi menjadi data latih (80%) dan data uji (20%):

```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

Untuk mengatasi ketidakseimbangan kelas pada data latih, digunakan teknik SMOTE (Synthetic Minority Over-sampling Technique):

```python
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
```

SMOTE dipilih untuk mengatasi ketidakseimbangan kelas dengan menciptakan sampel sintetis dari kelas minoritas (Survived=1), sehingga model dapat belajar dengan lebih baik dari kedua kelas.

## Modeling

Pada proyek ini, beberapa algoritma klasifikasi diterapkan dan dibandingkan untuk memprediksi kelangsungan hidup penumpang Titanic:

### 1. Logistic Regression

```python
LogisticRegression(max_iter=1000, class_weight='balanced')
```

**Kelebihan**:
- Mudah diinterpretasi
- Efisien secara komputasional
- Memberikan probabilitas yang terkalibrasi dengan baik

**Kekurangan**:
- Mengasumsikan hubungan linear antara fitur dan log-odds dari target
- Kurang mampu menangkap hubungan kompleks dalam data

### 2. Random Forest Classifier

```python
RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight='balanced')
```

**Kelebihan**:
- Mampu menangkap hubungan non-linear
- Tahan terhadap overfitting
- Memberikan feature importance yang berguna untuk interpretasi

**Kekurangan**:
- Lebih komputasional intensif dibandingkan Logistic Regression
- Kurang interpretable dibandingkan model parametrik

### 3. Gradient Boosting Classifier

```python
GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
```

**Kelebihan**:
- Performa tinggi melalui teknik boosting
- Mampu menangkap pola kompleks
- Tahan terhadap outlier

**Kekurangan**:
- Lebih rentan terhadap overfitting
- Lebih komputasional intensif
- Membutuhkan tuning parameter yang hati-hati

### 4. XGBoost Classifier

```python
XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, subsample=0.8, 
              colsample_bytree=0.8, use_label_encoder=False, eval_metric='logloss', random_state=42)
```

**Kelebihan**:
- Salah satu algoritma terbaik untuk klasifikasi
- Mengimplementasikan regularisasi untuk mencegah overfitting
- Efisien secara komputasional melalui komputasi paralel

**Kekurangan**:
- Lebih banyak hyperparameter untuk di-tuning
- Kurang intuitif untuk diinterpretasi

### Pemilihan Model Terbaik

Setelah melakukan evaluasi terhadap keempat model, Random Forest Classifier dipilih sebagai model terbaik dengan performa sebagai berikut:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest Classifier | 0.84 | 0.82 | 0.83 | 0.82 |

Random Forest Classifier dipilih sebagai model terbaik karena beberapa alasan:

1. **Akurasi Tinggi**: Model ini mencapai akurasi 84%, yang merupakan salah satu nilai tertinggi dibandingkan model lainnya.
2. **Keseimbangan Precision-Recall**: Model ini menawarkan keseimbangan yang baik antara precision (82%) dan recall (83%), yang penting untuk kasus di mana kita perlu meminimalkan baik false positives maupun false negatives.
3. **Feature Importance**: Random Forest menyediakan feature importance yang jelas, memungkinkan kita untuk mengidentifikasi faktor-faktor yang paling berpengaruh terhadap kelangsungan hidup.
4. **Ketahanan terhadap Overfitting**: Dengan kombinasi parameter yang tepat (max_depth=8), model ini mampu menangkap pola kompleks dalam data tanpa overfitting.

Model ini kemudian disimpan untuk digunakan dalam prediksi masa depan:

```python
joblib.dump(models["Random Forest Classifier"], 'model/best_model.pkl')
```

## Evaluation

Untuk mengevaluasi performa model klasifikasi dalam memprediksi kelangsungan hidup penumpang Titanic, beberapa metrik evaluasi yang digunakan adalah:

### 1. Accuracy

Accuracy mengukur proporsi prediksi yang benar (baik positif maupun negatif) dari total prediksi:

$$ \text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} $$

di mana:
- TP (True Positive): Jumlah penumpang yang diprediksi selamat dan memang selamat.
- TN (True Negative): Jumlah penumpang yang diprediksi tidak selamat dan memang tidak selamat.
- FP (False Positive): Jumlah penumpang yang diprediksi selamat tetapi sebenarnya tidak selamat.
- FN (False Negative): Jumlah penumpang yang diprediksi tidak selamat tetapi sebenarnya selamat.

### 2. Precision

Precision mengukur proporsi prediksi positif yang benar dari semua prediksi positif:

$$ \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} $$

Precision tinggi menunjukkan bahwa ketika model memprediksi penumpang akan selamat, prediksi tersebut kemungkinan besar benar.

### 3. Recall

Recall mengukur proporsi positif aktual yang diprediksi dengan benar:

$$ \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} $$

Recall tinggi menunjukkan bahwa model berhasil mengidentifikasi sebagian besar penumpang yang sebenarnya selamat.

### 4. F1-Score

F1-Score adalah rata-rata harmonik dari precision dan recall, memberikan keseimbangan antara keduanya:

$$ \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$

F1-Score tinggi menunjukkan keseimbangan yang baik antara precision dan recall.

### Hasil Evaluasi Model

Berikut adalah hasil evaluasi keempat model klasifikasi yang digunakan:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest Classifier | 0.84 | 0.82 | 0.83 | 0.82 |
| XGBoost Classifier | 0.82 | 0.82 | 0.79 | 0.80 |
| Gradient Boosting Classifier | 0.81 | 0.79 | 0.79 | 0.79 |
| Logistic Regression | 0.81 | 0.79 | 0.79 | 0.79 |

Dari hasil evaluasi, **Random Forest Classifier** menunjukkan performa terbaik dengan akurasi 84%, precision 82%, recall 83%, dan F1-Score 82%. Model ini mampu menyeimbangkan kemampuan untuk mengidentifikasi penumpang yang selamat (recall tinggi) dan menghasilkan prediksi yang akurat ketika memprediksi penumpang akan selamat (precision tinggi).

### Feature Importance

Analisis feature importance dari model Random Forest menunjukkan tiga fitur teratas yang paling berpengaruh terhadap prediksi kelangsungan hidup penumpang Titanic:

1. **Sex (0.32)**: Jenis kelamin merupakan faktor paling penting dalam menentukan kelangsungan hidup, dengan penumpang perempuan memiliki kemungkinan selamat yang jauh lebih tinggi dibandingkan laki-laki.
2. **Fare (0.22)**: Tarif tiket yang lebih tinggi (yang umumnya terkait dengan kelas tiket yang lebih tinggi) berhubungan dengan kemungkinan selamat yang lebih besar.
3. **Age (0.19)**: Usia penumpang juga memainkan peran penting, dengan penumpang yang lebih muda memiliki kemungkinan selamat yang lebih tinggi.

Hasil ini konsisten dengan fakta historis bahwa selama evakuasi Titanic, prioritas diberikan kepada wanita dan anak-anak ("women and children first"), sementara penumpang dari kelas yang lebih tinggi (yang membayar tarif lebih tinggi) memiliki akses yang lebih baik ke sekoci penyelamat.

Wawasan ini dapat membantu dalam pengembangan protokol keselamatan yang lebih baik dan pengambilan keputusan terkait alokasi sumber daya keselamatan dalam situasi darurat.

## Referensi

[1] Britannica, The Editors of Encyclopaedia. "Titanic". Encyclopedia Britannica, 7 Apr. 2023, https://www.britannica.com/topic/Titanic. Diakses 5 Mei 2025.
