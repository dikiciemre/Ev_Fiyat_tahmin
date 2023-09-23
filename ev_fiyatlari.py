
"""
Metrekare,Oda Sayısı,Konum,Ev Yaşı,Ev Fiyatı
150,3,Merkez,10,250000
200,4,Çevre,5,320000
120,2,Merkez,15,200000
180,3,Çevre,8,290000
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Örnek veri seti
data = {'Metrekare': [150, 200, 120, 180,230,300,100],
        'Oda Sayısı': [3, 4, 2, 3,5,6,17],
        'Konum': ["Merkez", "Çevre", "Merkez", "Çevre","Merkez","Çevre","Merkez"],
        'Ev Yaşı': [10, 5, 15, 8,3,13,19],
        'Ev Fiyatı': [250000, 320000, 200000, 290000,310000,400000,140000]}

# Veri setini Pandas DataFrame'e dönüştürme
df = pd.DataFrame(data)

# Kategorik verileri nümerik değerlere dönüştürme
label_encoder = LabelEncoder()
df['Konum'] = label_encoder.fit_transform(df['Konum'])

# Bağımsız değişkenler ve bağımlı değişkeni ayırma
X = df.drop('Ev Fiyatı', axis=1)
y = df['Ev Fiyatı']

# Veriyi eğitim ve test setlerine böleme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Tahminler yapma
y_pred = model.predict(X_test)

# Model performansını değerlendirme
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Ortalama Kare Hata (MSE):', mse)
print('R-Kare (R^2):', r2)

# Yeni bir evin fiyatını tahmin etme örneği
new_data = pd.DataFrame({'Metrekare': [170],
                         'Oda Sayısı': [3],
                         'Konum': ['Çevre'],
                         'Ev Yaşı': [8]})

# Kategorik veriyi nümerik değere dönüştürme (label_encoder'ı tekrar kullanma)
new_data['Konum'] = label_encoder.transform(new_data['Konum'])

# Tahmin yapma
new_price = model.predict(new_data)

print('Yeni evin tahmini fiyatı:', new_price[0])
