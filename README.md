# NorthwindAutoencoderRecommender-NewProductPurchasePotential-

# 🛍️ Yeni Ürün Satın Alma Potansiyeli Tahmini (FastAPI + AutoEncoder)

Bu proje, Northwind veritabanındaki müşteri satın alma geçmişlerini analiz ederek yeni çıkan ürün kategorilerine karşı potansiyel ilgilerini tahmin etmeyi amaçlar.

## 📌 Amaç

Müşterilerin geçmişte harcama yaptıkları kategori dağılımını analiz ederek:
- **Satın alma potansiyelini** tahmin eden bir sinir ağı modeli (AutoEncoder) geliştirildi.
- Öneri sistemi API formatında sunuldu (FastAPI ile).

## 🧠 Kullanılan Teknolojiler

- Python, Pandas, NumPy
- FastAPI
- TensorFlow
- PostgreSQL (Northwind veritabanı)
- Matplotlib / Seaborn (Görselleştirme)

## 🔍 Proje Özeti

- `Products`, `Categories`, `Order Details`, `Orders` tabloları birleştirildi.
- Müşteri-kategori harcamaları üzerinden bir **pivot tablo** oluşturuldu.
- Bu tablo standardize edilerek **AutoEncoder** ile yeniden üretildi.
- Eksik kategorilere olan potansiyel harcama eğilimleri tahmin edildi.
- Tahminler, FastAPI üzerinden `POST /predict` endpoint’i ile alınabilir.

## 📊 Isı Haritaları

**En Çok Harcayan Müşteriler:**

![top_heatmap](https://github.com/user-attachments/assets/e2e70596-b72e-4900-9853-38a7a98edb01)


**En Az Harcayan Müşteriler:**

![bottom_heatmap](https://github.com/user-attachments/assets/538080d7-1e4c-4c27-a312-c3487c3009e7)


**FastAPI sunucusunu başlat**
uvicorn problem3:app --reload

**API arayüzüne git**
http://localhost:8000/docs




<sub>📌 Dipnot:Bu çalışma Turkcell GYK programı kapsamında ödev olarak yapılmıştır. 
