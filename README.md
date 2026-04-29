# E-Commerce Analytics Dashboard 🛒

Dashboard analisis data e-commerce Brazil menggunakan dataset Olist (2016–2018), mencakup analisis revenue, kategori produk, segmentasi pelanggan (RFM), dan geospatial.

## Setup Environment - Anaconda

```bash
conda create --name ecommerce-analysis python=3.11.5
conda activate ecommerce-analysis
pip install -r requirements.txt
```

## Setup Environment - Shell/Terminal

```bash
mkdir submission
cd submission
pipenv install
pipenv shell
pip install -r requirements.txt
```

## Run Streamlit App

```bash
streamlit run dashboard/dashboard.py
```