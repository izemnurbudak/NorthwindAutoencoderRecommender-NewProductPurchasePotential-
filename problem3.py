import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from sqlalchemy import create_engine
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi.staticfiles import StaticFiles
import os

os.makedirs("static", exist_ok=True)

# PostgreSQL bağlantı bilgileri
user = "postgres"
password = "0000"
host = "localhost"
port = "5432"
database = "northwind"

# SQLAlchemy engine
engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}')

# SQL sorgusu
query = """
SELECT 
    o.customer_id,
    c.category_name,
    SUM(od.unit_price * od.quantity) AS total_amount,
    COUNT(DISTINCT o.order_id) AS order_count
FROM orders o
JOIN order_details od ON o.order_id = od.order_id
JOIN products p ON od.product_id = p.product_id
JOIN categories c ON p.category_id = c.category_id
GROUP BY o.customer_id, c.category_name
"""

try:
    df = pd.read_sql_query(query, engine)
    print("Veri başarıyla çekildi!")
except Exception as e:
    print(f"Veri çekme hatası: {e}")
    raise

# Pivot table
customer_category_matrix = df.pivot_table(
    index='customer_id',
    columns='category_name',
    values='total_amount',
    fill_value=0
)

print(customer_category_matrix)


# Standardizasyon
scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_category_matrix)

# AutoEncoder modeli
input_dim = scaled_features.shape[1]

def build_autoencoder():
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(32, activation='relu')(input_layer)
    encoded = layers.Dense(16, activation='relu')(encoded)
    latent = layers.Dense(8, activation='relu')(encoded)
    decoded = layers.Dense(16, activation='relu')(latent)
    decoded = layers.Dense(32, activation='relu')(decoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)
    autoencoder = models.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# Model eğitimi
X_train, X_test = train_test_split(scaled_features, test_size=0.2, random_state=42)
autoencoder = build_autoencoder()
history = autoencoder.fit(X_train, X_train, epochs=100, batch_size=32, shuffle=True, validation_data=(X_test, X_test))

# Tahmin fonksiyonu
def predict_category_likelihood(customer_id, category_name):
    try:
        customer_profile = customer_category_matrix.loc[customer_id].values.reshape(1, -1)
        customer_profile_scaled = scaler.transform(customer_profile)
        predicted_profile = autoencoder.predict(customer_profile_scaled)
        predicted_profile = scaler.inverse_transform(predicted_profile)
        category_index = list(customer_category_matrix.columns).index(category_name)
        return predicted_profile[0][category_index]
    except Exception as e:
        print(f"Tahmin hatası: {e}")
        return None

# FastAPI uygulaması
app = FastAPI(
    title="Yeni Ürün Satın Alma Potansiyeli",
    description="""
🔍 **Proje Amacı**
- Bu API, müşterilerin geçmiş alışveriş davranışlarını analiz ederek, henüz satın almadıkları kategorilere olan potansiyel ilgilerini tahmin eder. Bu amaçla **AutoEncoder tabanlı bir derin öğrenme mimarisi** kullanılmıştır.

🔢 **Model Yaklaşımı (AutoEncoder)**
- AutoEncoder, girişteki kullanıcı-kategori harcama vektörünü alarak onu sıkıştırılmış bir temsil (latent space) üzerinden yeniden üretmeye çalışır. Böylece eksik veya gözlemlenmemiş kategoriler için tahmini harcama düzeyleri öğrenilir.

🔧 **Teknoloji**
- Deep Learning (AutoEncoder)
- StandardScaler (veri normalizasyonu)
- Matrix Completion mantığıyla eksik kategorilere ilgi tahmini

📦 **Veri Kaynağı**
- Northwind veritabanı (orders, order_details, products, categories)

🎯 **Kullanım Senaryosu**
- Pazarlama ekiplerinin kişiselleştirilmiş kampanya oluşturması
- Yeni ürün öneri sistemi oluşturulması
- Kategori bazlı müşteri segmentasyonu


Bu sistem, müşterilerin geçmiş sipariş verilerine dayanarak oluşturulan pivot tablo aracılığıyla 
kategori bazında harcama davranışlarını analiz eder. 

Elde edilen veriler görsel olarak ısı haritası ile sunulur. 

📊 **Isı Haritaları**
<p><strong>En Çok Harcama Yapan 5 Müşterinin En Çok Harcama Yaptığı 5 Kategori:</strong></p>
<img src="/static/top_heatmap.png" alt="Top Harcayan Harita" width="600"/>

<p><strong>En Az Harcama Yapan 5 Müşterinin En Az Harcama Yaptığı 5 Kategori:</strong></p>
<img src="/static/bottom_heatmap.png" alt="Az Harcayan Harita" width="600"/>

Bu analiz doğrultusunda, her bir kategori için potansiyel harcama aralıkları belirlenir ve bu 
bilgiler öneri sisteminin temelini oluşturur.

🔎 Tahmin edilen satın alma potansiyelinin yorumlaması:

- 🔥 Very high potential to buy  → if likelihood > 1500  
- ✅ High potential to buy       → if likelihood > 900  
- ⚠️ Moderate potential to buy   → if likelihood > 400  
- ❌ Low potential to buy        → otherwise  
"""
,
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    customer_id: str
    category_name: str

@app.post("/predict")
def predict(request: PredictionRequest):
    customer_id = request.customer_id
    category_name = request.category_name

    if customer_id not in customer_category_matrix.index:
        raise HTTPException(status_code=404, detail="Customer not found")
    if category_name not in customer_category_matrix.columns:
        raise HTTPException(status_code=404, detail="Category not found")

    likelihood = predict_category_likelihood(customer_id, category_name)
    if likelihood is None:
        raise HTTPException(status_code=500, detail="Prediction error")
    
    if likelihood > 1500:
        result = "🔥 Very high potential to buy"
    elif likelihood > 1000:
        result = "✅ High potential to buy"
    elif likelihood > 500:
        result = "⚠️ Moderate potential to buy"
    else:
        result = "❌ Low potential to buy"

    return {
        "customer_id": customer_id,
        "category_name": category_name,
        "likelihood": float(likelihood),
        "result": result    
    }

# Eğitim kaybı görselleştirmesi
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

def save_heatmap(df, filename, title="Heatmap", cmap="YlGnBu"):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt=".1f", cmap=cmap, linewidths=0.3)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

app.mount("/static", StaticFiles(directory="static"), name="static")

def plot_top_bottom_heatmaps(pivot_table, top_n=5):
    top_customers = pivot_table.sum(axis=1).sort_values(ascending=False).head(top_n).index
    top_categories = pivot_table.sum(axis=0).sort_values(ascending=False).head(top_n).index
    bottom_customers = pivot_table.sum(axis=1).sort_values().head(top_n).index
    bottom_categories = pivot_table.sum(axis=0).sort_values().head(top_n).index

    top_matrix = pivot_table.loc[top_customers, top_categories]
    save_heatmap(top_matrix, "static/top_heatmap.png", "En Çok Harcama Yapılan Kategoriler ve Müşteriler", cmap="YlGnBu")

    bottom_matrix = pivot_table.loc[bottom_customers, bottom_categories]
    save_heatmap(bottom_matrix, "static/bottom_heatmap.png", "En Az Harcama Yapılan Kategoriler ve Müşteriler", cmap="YlOrBr")

# En son çağır
plot_top_bottom_heatmaps(customer_category_matrix, top_n=5)