import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import os
from st_aggrid import AgGrid, GridOptionsBuilder
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats

# ─── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #00521f 0%, #00B14F 100%);
    }
    [data-testid="stSidebar"] * { color: white !important; }

    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-top: 4px solid #00B14F;
        height: 110px;
    }
    .kpi-label { font-size: 11px; color: #888; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-value { font-size: 24px; font-weight: 800; color: #1a1a1a; margin: 4px 0; }
    .kpi-growth-pos { font-size: 12px; color: #00B14F; font-weight: 700; }
    .kpi-growth-neg { font-size: 12px; color: #FF5252; font-weight: 700; }

    .section-title {
        font-size: 15px; font-weight: 700; color: #00521f;
        padding: 8px 0 4px 0;
        border-bottom: 2px solid #00B14F;
        margin-bottom: 12px;
    }

    .insight-box {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border-left: 4px solid #00B14F;
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 8px;
        font-size: 12px;
        color: #166534;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: white;
        border-radius: 10px;
        padding: 4px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        color: #666;
        padding: 8px 20px;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background: #00B14F !important;
        color: white !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
            
/* Tabel styling */
[data-testid="stDataFrame"] {
    background: white !important;
}
[data-testid="stDataFrame"] tr:nth-child(even) {
    background: #f0fdf4 !important;
}
[data-testid="stDataFrame"] th {
    background: #00B14F !important;
    color: white !important;
    font-weight: 700 !important;
}
[data-testid="stDataFrame"] td {
    color: #1a1a1a !important;
}

</style>
""", unsafe_allow_html=True)

# ─── LOAD DATA ─────────────────────────────────────────────
@st.cache_data
def load_data():
    main_data = pd.read_csv('dashboard/main_data.csv')
    rfm_data = pd.read_csv('dashboard/rfm_data.csv')
    revenue_state = pd.read_csv('dashboard/revenue_state.csv')
    main_data['order_purchase_timestamp'] = pd.to_datetime(main_data['order_purchase_timestamp'])
    return main_data, rfm_data, revenue_state

main_data, rfm_data, revenue_state = load_data()
def show_table(df, height=350):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(resizable=True, filterable=True, sortable=True)
    gb.configure_grid_options(rowHeight=40, headerHeight=45)
    gridOptions = gb.build()
    AgGrid(
        df,
        gridOptions=gridOptions,
        height=height,
        theme='alpine',
        fit_columns_on_grid_load=True
    )

# ─── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:10px 0'>
        <div style='font-size:40px'>🛒</div>
        <div style='font-size:18px; font-weight:800'>E-Commerce</div>
        <div style='font-size:12px; opacity:0.8'>Analytics Dashboard</div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.3)'>
    """, unsafe_allow_html=True)

    # Tahun
    st.markdown("<div style='font-size:11px; font-weight:700; opacity:0.9; margin-bottom:4px'>📅 TAHUN</div>", unsafe_allow_html=True)
    years = sorted(main_data['year'].dropna().unique().astype(int))
    selected_years = st.multiselect("", years, default=years, key='years', label_visibility='collapsed')

    st.markdown("<hr style='border-color:rgba(255,255,255,0.3)'>", unsafe_allow_html=True)

    # Rentang tanggal
    st.markdown("<div style='font-size:11px; font-weight:700; opacity:0.9; margin-bottom:4px'>📅 RENTANG TANGGAL</div>", unsafe_allow_html=True)
    min_date = main_data['order_purchase_timestamp'].min().date()
    max_date = main_data['order_purchase_timestamp'].max().date()
    date_range = st.date_input("", value=(min_date, max_date),
        min_value=min_date, max_value=max_date,
        key='date_range', label_visibility='collapsed')

    st.markdown("<hr style='border-color:rgba(255,255,255,0.3)'>", unsafe_allow_html=True)

    # Payment type
    st.markdown("<div style='font-size:11px; font-weight:700; opacity:0.9; margin-bottom:4px'>💳 PAYMENT TYPE</div>", unsafe_allow_html=True)
    payment_types = sorted(main_data['payment_type'].dropna().unique())
    selected_payments = st.multiselect("", payment_types, default=payment_types, key='payments', label_visibility='collapsed')

    st.markdown("<hr style='border-color:rgba(255,255,255,0.3)'>", unsafe_allow_html=True)

    # Kategori
    st.markdown("<div style='font-size:11px; font-weight:700; opacity:0.9; margin-bottom:4px'>📦 KATEGORI</div>", unsafe_allow_html=True)
    categories = sorted(main_data['product_category_name_english'].dropna().unique())
    selected_categories = st.multiselect("", categories, default=categories, key='categories', label_visibility='collapsed')

    st.markdown("<hr style='border-color:rgba(255,255,255,0.3)'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:11px; opacity:0.8; text-align:center'>
        📊 Olist Brazil Dataset<br>2016 - 2018
    </div>""", unsafe_allow_html=True)

# ─── FILTER DATA ───────────────────────────────────────────
if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

filtered_df = main_data[
    (main_data['year'].isin(selected_years)) &
    (main_data['product_category_name_english'].isin(selected_categories)) &
    (main_data['payment_type'].isin(selected_payments)) &
    (main_data['order_purchase_timestamp'].dt.date >= start_date) &
    (main_data['order_purchase_timestamp'].dt.date <= end_date)
]

def calc_growth(metric_col, agg='sum'):
    if len(selected_years) == 0: return 0
    curr_year = max(selected_years)
    prev_year = curr_year - 1
    curr = main_data[main_data['year'] == curr_year][metric_col]
    prev = main_data[main_data['year'] == prev_year][metric_col]
    curr_val = curr.sum() if agg == 'sum' else curr.nunique()
    prev_val = prev.sum() if agg == 'sum' else prev.nunique()
    if prev_val == 0: return 0
    return ((curr_val - prev_val) / prev_val) * 100

rev_growth = calc_growth('payment_value', 'sum')
ord_growth = calc_growth('order_id', 'nunique')
cust_growth = calc_growth('customer_unique_id', 'nunique')

total_revenue = filtered_df['payment_value'].sum()
total_orders = filtered_df['order_id'].nunique()
total_customers = filtered_df['customer_unique_id'].nunique()
avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
top_payment = filtered_df['payment_type'].value_counts().index[0] if len(filtered_df) > 0 else 'N/A'

def growth_html(val):
    arrow = "▲" if val >= 0 else "▼"
    cls = "kpi-growth-pos" if val >= 0 else "kpi-growth-neg"
    return f"<span class='{cls}'>{arrow} {abs(val):.1f}% vs last year</span>"

# ─── HEADER ────────────────────────────────────────────────
st.markdown("""
<div style='padding:12px 0 8px 0'>
    <span style='font-size:24px; font-weight:800; color:#00521f'>🛒 E-Commerce Analytics</span>
    <span style='font-size:13px; color:#888; margin-left:12px'>Olist Brazil | 2016-2018</span>
</div>""", unsafe_allow_html=True)

# ─── KPI CARDS (selalu tampil di semua tab) ─────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""<div class='kpi-card'>
        <div class='kpi-label'>💰 Total Revenue</div>
        <div class='kpi-value'>R$ {total_revenue/1e6:.2f}M</div>
        {growth_html(rev_growth)}</div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class='kpi-card'>
        <div class='kpi-label'>📦 Total Orders</div>
        <div class='kpi-value'>{total_orders:,}</div>
        {growth_html(ord_growth)}</div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class='kpi-card'>
        <div class='kpi-label'>👥 Total Customers</div>
        <div class='kpi-value'>{total_customers:,}</div>
        {growth_html(cust_growth)}</div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class='kpi-card'>
        <div class='kpi-label'>🧾 Avg Order Value</div>
        <div class='kpi-value'>R$ {avg_order_value:.0f}</div>
        <span class='kpi-growth-pos'>per transaksi</span></div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""<div class='kpi-card'>
        <div class='kpi-label'>💳 Top Payment</div>
        <div class='kpi-value' style='font-size:16px'>{top_payment.replace('_',' ').title()}</div>
        <span class='kpi-growth-pos'>most used method</span></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 Overview", "🛍️ Product Analysis", "👥 Customer", "🗺️ Geospatial",
    "📉 Forecasting", "🤖 ML Churn", "📊 Adv. Viz", "🧪 A/B Testing"
])

segment_colors = {
    'Champions':'#00521f','Loyal Customers':'#00B14F',
    'Potential Loyalists':'#1DE9B6','At Risk':'#FFB300','Lost Customers':'#FF5252'
}

# ════════════════════════════════════════════
# TAB 1: OVERVIEW
# ════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-title'>📈 Revenue & Transaksi Trend</div>", unsafe_allow_html=True)

    monthly = filtered_df.groupby(['year','month']).agg(
        revenue=('payment_value','sum'),
        orders=('order_id','nunique')
    ).reset_index()
    monthly['date'] = pd.to_datetime(monthly[['year','month']].assign(day=1))
    monthly = monthly.sort_values('date')

    col1, col2, col3 = st.columns([2,2,1])

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly['date'], y=monthly['revenue'],
            fill='tozeroy', fillcolor='rgba(0,177,79,0.15)',
            line=dict(color='#00B14F', width=2.5), name='Revenue'
        ))
        fig.update_layout(
            title=dict(text='Monthly Revenue', font=dict(size=14, color='#00521f', family='Arial Black')),
            height=220, margin=dict(t=35,b=30,l=40,r=10),
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(showgrid=False, tickfont=dict(color='#1a1a1a', size=11)),
            yaxis=dict(gridcolor='#f0f0f0', tickfont=dict(color='#1a1a1a', size=11)),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        max_month = monthly.loc[monthly['revenue'].idxmax()]
        st.markdown(f"""<div class='insight-box'>
            💡 Revenue tertinggi terjadi pada <b>{max_month['date'].strftime('%B %Y')}</b> 
            sebesar <b>R$ {max_month['revenue']/1e6:.2f}M</b>. 
            Kemungkinan dipicu oleh event Black Friday.
        </div>""", unsafe_allow_html=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=monthly['date'], y=monthly['orders'],
            marker_color='#00521f', name='Orders'
        ))
        fig2.update_layout(
            title=dict(text='Monthly Orders', font=dict(size=14, color='#00521f', family='Arial Black')),
            height=220, margin=dict(t=35,b=30,l=40,r=10),
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(showgrid=False, tickfont=dict(color='#1a1a1a', size=11)),
            yaxis=dict(gridcolor='#f0f0f0', tickfont=dict(color='#1a1a1a', size=11)),
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)
        avg_orders = monthly['orders'].mean()
        max_orders = monthly['orders'].max()
        st.markdown(f"""<div class='insight-box'>
            💡 Rata-rata <b>{avg_orders:,.0f} orders/bulan</b>. 
            Puncak transaksi mencapai <b>{max_orders:,} orders</b> dalam satu bulan.
        </div>""", unsafe_allow_html=True)

    with col3:
        payment = filtered_df.groupby('payment_type')['payment_value'].sum().reset_index()
        fig3 = go.Figure(go.Pie(
            values=payment['payment_value'],
            labels=payment['payment_type'],
            hole=0.5,
            marker_colors=['#00521f','#00B14F','#1DE9B6','#00BFA5'],
            textinfo='percent', textfont_size=10
        ))
        fig3.update_layout(
            title=dict(text='Payment Mix',font=dict(size=14, color='#00521f', family='Arial Black')),
            height=220, margin=dict(t=35,b=10,l=10,r=10),
            paper_bgcolor='white',
            legend=dict(
                  font=dict(size=10, color='#1a1a1a'),
                  bgcolor='white',
                  bordercolor='#e0e0e0',
                  borderwidth=1
            )
        )
        st.plotly_chart(fig3, use_container_width=True)
        top_pct = payment['payment_value'].max()/payment['payment_value'].sum()*100
        st.markdown(f"""<div class='insight-box'>
            💡 <b>{top_payment.replace('_',' ').title()}</b> mendominasi 
            <b>{top_pct:.1f}%</b> dari total revenue.
        </div>""", unsafe_allow_html=True)

    # Tabel ringkasan bulanan
    st.markdown("<div class='section-title'>📋 Data Tren Bulanan</div>", unsafe_allow_html=True)
    monthly_display = monthly.copy()
    monthly_display['date'] = monthly_display['date'].dt.strftime('%Y-%m')
    monthly_display['revenue'] = monthly_display['revenue'].apply(lambda x: f'R$ {x:,.2f}')
    monthly_display = monthly_display[['date','revenue','orders']].rename(columns={
        'date':'Bulan','revenue':'Revenue','orders':'Orders'
    })
    show_table(monthly_display, height=250)

# ════════════════════════════════════════════
# TAB 2: PRODUCT ANALYSIS
# ════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>🛍️ Analisis Kategori Produk</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([3,2])

    with col1:
        top_cat = filtered_df.groupby('product_category_name_english')['payment_value'].sum()\
            .reset_index().sort_values('payment_value', ascending=True).tail(10)
        fig4 = go.Figure(go.Bar(
            x=top_cat['payment_value'],
            y=top_cat['product_category_name_english'],
            orientation='h',
            marker=dict(color=top_cat['payment_value'], colorscale=[[0,'#c8f5d8'],[1,'#00521f']]),
            text=top_cat['payment_value'].apply(lambda x: f'R${x/1e6:.2f}M'),
            textposition='outside', textfont=dict(size=10, color='#1a1a1a')
        ))
        fig4.update_layout(
            title=dict(text='Top 10 Kategori by Revenue', font=dict(size=14, color='#00521f', family='Arial Black')),
            height=300, margin=dict(t=35,b=20,l=10,r=100),
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(showgrid=False, tickfont=dict(color='#1a1a1a', size=11)),
            yaxis=dict(gridcolor='#f0f0f0', tickfont=dict(color='#1a1a1a', size=11)),
            showlegend=False
        )
        st.plotly_chart(fig4, use_container_width=True)
        top1 = top_cat.iloc[-1]
        st.markdown(f"""<div class='insight-box'>
            💡 Kategori <b>{top1['product_category_name_english']}</b> adalah revenue driver utama 
            dengan total <b>R$ {top1['payment_value']/1e6:.2f}M</b>. 
            Fokus pada stok dan promosi kategori ini untuk memaksimalkan revenue.
        </div>""", unsafe_allow_html=True)

    with col2:
        # Revenue per bulan per top 5 kategori
        top5_cats = filtered_df.groupby('product_category_name_english')['payment_value']\
            .sum().nlargest(5).index
        cat_monthly = filtered_df[filtered_df['product_category_name_english'].isin(top5_cats)]\
            .groupby(['product_category_name_english','year','month'])['payment_value'].sum()\
            .reset_index()
        cat_monthly['date'] = pd.to_datetime(cat_monthly[['year','month']].assign(day=1))
        fig5 = px.line(cat_monthly, x='date', y='payment_value',
                       color='product_category_name_english',
                       color_discrete_sequence=['#00521f','#00B14F','#1DE9B6','#00BFA5','#26C6DA'])
        fig5.update_layout(
            title=dict(text='Tren Top 5 Kategori per Bulan', 
                       font=dict(size=14, color='#00521f', family='Arial Black')),
            height=300, margin=dict(t=35,b=20,l=10,r=10),
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(showgrid=False, tickfont=dict(color='#1a1a1a', size=11)),
            yaxis=dict(gridcolor='#f0f0f0', tickfont=dict(color='#1a1a1a', size=11)),
            legend=dict(font=dict(size=9, color='#1a1a1a'), title='', bgcolor='white', bordercolor='#e0e0e0', borderwidth=1)
        )
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown(f"""<div class='insight-box'>
            💡 Semua kategori menunjukkan tren <b>naik</b> dari 2017 ke 2018. 
            Lonjakan tajam terlihat di <b>Nov 2017</b> (Black Friday effect).
        </div>""", unsafe_allow_html=True)

    # Tabel kategori interaktif
    st.markdown("<div class='section-title'>📋 Tabel Kategori — Sortable</div>", unsafe_allow_html=True)
    cat_table = filtered_df.groupby('product_category_name_english').agg(
        Total_Revenue=('payment_value','sum'),
        Total_Orders=('order_id','nunique'),
        Avg_Order_Value=('payment_value','mean'),
        Total_Customers=('customer_unique_id','nunique')
    ).reset_index().sort_values('Total_Revenue', ascending=False)
    cat_table.columns = ['Kategori','Total Revenue (R$)','Total Orders','Avg Order Value (R$)','Total Customers']
    cat_table['Total Revenue (R$)'] = cat_table['Total Revenue (R$)'].round(2)
    cat_table['Avg Order Value (R$)'] = cat_table['Avg Order Value (R$)'].round(2)
    show_table(cat_table)

# ════════════════════════════════════════════
# TAB 3: CUSTOMER (RFM)
# ════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>👥 RFM Customer Segmentation</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        seg_count = rfm_data['Segment'].value_counts().reset_index()
        seg_count.columns = ['Segment','Count']
        fig6 = go.Figure(go.Pie(
            values=seg_count['Count'], labels=seg_count['Segment'],
            hole=0.5,
            marker_colors=[segment_colors.get(s,'gray') for s in seg_count['Segment']],
            textinfo='percent+label', textfont_size=10
        ))
        fig6.update_layout(
            title=dict(text='Distribusi Segmen', font=dict(size=14, color='#00521f', family='Arial Black')),
            height=300, margin=dict(t=35,b=10,l=10,r=10),
            paper_bgcolor='white', showlegend=False
        )
        st.plotly_chart(fig6, use_container_width=True)
        top_seg = seg_count.iloc[0]
        st.markdown(f"""<div class='insight-box'>
            💡 Segmen terbesar adalah <b>{top_seg['Segment']}</b> ({top_seg['Count']:,} pelanggan). 
            Fokus konversi segmen ini ke Loyal Customers dengan program loyalitas.
        </div>""", unsafe_allow_html=True)

    with col2:
        seg_monetary = rfm_data.groupby('Segment')['Monetary'].mean().reset_index()
        order_seg = ['Champions','Loyal Customers','Potential Loyalists','At Risk','Lost Customers']
        seg_monetary['Segment'] = pd.Categorical(seg_monetary['Segment'], categories=order_seg, ordered=True)
        seg_monetary = seg_monetary.sort_values('Segment')
        fig7 = go.Figure(go.Bar(
            x=seg_monetary['Segment'], y=seg_monetary['Monetary'],
            marker_color=[segment_colors.get(s,'gray') for s in seg_monetary['Segment']],
            text=seg_monetary['Monetary'].apply(lambda x: f'R${x:.0f}'),
            textposition='outside', textfont=dict(size=10, color='#1a1a1a')
        ))
        fig7.update_layout(
            title=dict(text='Avg Monetary per Segmen', font=dict(size=14, color='#00521f', family='Arial Black')),
            height=300, margin=dict(t=35,b=20,l=10,r=10),
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(showgrid=False, tickfont=dict(color='#1a1a1a', size=9)),
            yaxis=dict(gridcolor='#f0f0f0', tickfont=dict(color='#1a1a1a', size=9)),
            showlegend=False
        )
        st.plotly_chart(fig7, use_container_width=True)
        champions_val = seg_monetary[seg_monetary['Segment']=='Champions']['Monetary'].values[0]
        lost_val = seg_monetary[seg_monetary['Segment']=='Lost Customers']['Monetary'].values[0]
        st.markdown(f"""<div class='insight-box'>
            💡 Champions menghabiskan <b>R$ {champions_val:.0f}</b> rata-rata — 
            <b>{champions_val/lost_val:.1f}x</b> lebih besar dari Lost Customers.
        </div>""", unsafe_allow_html=True)

    with col3:
        seg_recency = rfm_data.groupby('Segment')['Recency'].mean().reset_index()
        seg_recency['Segment'] = pd.Categorical(seg_recency['Segment'], categories=order_seg, ordered=True)
        seg_recency = seg_recency.sort_values('Segment')
        fig8 = go.Figure(go.Bar(
            x=seg_recency['Segment'], y=seg_recency['Recency'],
            marker_color=[segment_colors.get(s,'gray') for s in seg_recency['Segment']],
            text=seg_recency['Recency'].apply(lambda x: f'{x:.0f}d'),
            textposition='outside', textfont=dict(size=10, color='#1a1a1a')
        ))
        fig8.update_layout(
            title=dict(text='Avg Recency per Segmen (hari)', font=dict(size=14, color='#00521f', family='Arial Black')),
            height=300, margin=dict(t=35,b=20,l=10,r=10),
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(showgrid=False, tickfont=dict(color='#1a1a1a', size=9)),
            yaxis=dict(gridcolor='#f0f0f0', tickfont=dict(color='#1a1a1a', size=9)),
            showlegend=False
        )
        st.plotly_chart(fig8, use_container_width=True)
        st.markdown(f"""<div class='insight-box'>
            💡 Champions terakhir beli paling baru. 
            At Risk & Lost Customers sudah lama tidak transaksi — perlu <b>win-back campaign</b>.
        </div>""", unsafe_allow_html=True)

    # Tabel RFM interaktif
    st.markdown("<div class='section-title'>📋 Tabel RFM — Sortable & Searchable</div>", unsafe_allow_html=True)
    order_seg = ['Champions','Loyal Customers','Potential Loyalists','At Risk','Lost Customers'] 
    st.markdown("<div style='font-size:11px; font-weight:700; color:#00521f; margin-bottom:4px'>🔍 FILTER SEGMEN</div>", unsafe_allow_html=True)
    seg_filter = st.multiselect("", options=order_seg, default=order_seg, key='seg_filter', label_visibility='collapsed')

    rfm_display = rfm_data[rfm_data['Segment'].isin(seg_filter)][
        ['customer_unique_id','Recency','Frequency','Monetary','Segment']
    ].copy()
    rfm_display['Monetary'] = rfm_display['Monetary'].round(2)
    rfm_display['Recency'] = rfm_display['Recency'].astype(int)
    rfm_display.columns = ['Customer ID','Recency (hari)','Frequency','Monetary (R$)','Segment']
    show_table(rfm_display)

    st.caption(f"Menampilkan {len(rfm_display):,} dari {len(rfm_data):,} pelanggan")

# ════════════════════════════════════════════
# TAB 4: GEOSPATIAL
# ════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-title'>🗺️ Geospatial Analysis — Revenue per State</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([3,2])

    with col1:
        map_path = "dashboard/brazil_map.html"
        if os.path.exists(map_path):
            with open(map_path, 'r', encoding='utf-8') as f:
                map_html = f.read()
            components.html(map_html, height=420)
        else:
            st.warning("Peta tidak ditemukan!")
        st.markdown(f"""<div class='insight-box'>
            💡 <b>São Paulo (SP)</b> mendominasi revenue Brazil dengan kontribusi terbesar. 
            Wilayah Tenggara Brazil adalah pusat ekonomi e-commerce terkuat.
        </div>""", unsafe_allow_html=True)

    with col2:
        top10_state = revenue_state.head(10)
        fig9 = go.Figure(go.Bar(
            x=top10_state['total_revenue'],
            y=top10_state['customer_state'],
            orientation='h',
            marker=dict(
                color=top10_state['total_revenue'],
                colorscale=[[0,'#c8f5d8'],[1,'#00521f']]
            ),
            text=top10_state['total_revenue'].apply(lambda x: f'R${x/1e6:.1f}M'),
            textposition='outside', textfont=dict(size=9, color='#1a1a1a')
        ))
        fig9.update_layout(
            title=dict(text='Top 10 State by Revenue', font=dict(size=14, color='#00521f', family='Arial Black')),
            height=420, margin=dict(t=35,b=20,l=10,r=80),
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(showgrid=False, tickfont=dict(color='#1a1a1a', size=9)),
            yaxis=dict(categoryorder='total ascending', tickfont=dict(color='#1a1a1a', size=9)),
            showlegend=False
        )
        st.plotly_chart(fig9, use_container_width=True)

    # Tabel state interaktif
    st.markdown("<div class='section-title'>📋 Tabel Revenue per State</div>", unsafe_allow_html=True)
    state_display = revenue_state.copy()
    state_display['total_revenue'] = state_display['total_revenue'].round(2)
    state_display['revenue_per_customer'] = (state_display['total_revenue'] / state_display['total_customers']).round(2)
    state_display.columns = ['State','Total Revenue (R$)','Total Orders','Total Customers','Revenue per Customer (R$)']
    show_table(state_display)

# ════════════════════════════════════════════
# TAB 5: TIME SERIES FORECASTING
# ════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-title'>📉 Time Series Forecasting — Revenue Bulanan</div>", unsafe_allow_html=True)

    monthly_ts = (
        filtered_df
        .groupby(pd.Grouper(key='order_purchase_timestamp', freq='MS'))['payment_value']
        .sum().reset_index()
        .rename(columns={'order_purchase_timestamp':'date','payment_value':'revenue'})
    )
    monthly_ts = monthly_ts[monthly_ts['revenue'] > 0].set_index('date')

    if len(monthly_ts) < 6:
        st.warning("Data terlalu sedikit untuk forecasting. Pilih rentang tahun yang lebih luas.")
    else:
        try:
            sp = min(6, len(monthly_ts) // 2)
            model_hw = ExponentialSmoothing(
                monthly_ts['revenue'], trend='add', seasonal='add', seasonal_periods=sp
            ).fit(optimized=True)
            steps = 3
            forecast_vals = model_hw.forecast(steps)
            forecast_idx  = pd.date_range(
                start=monthly_ts.index[-1] + pd.DateOffset(months=1), periods=steps, freq='MS'
            )
            if len(monthly_ts) >= 6:
                in_sample = model_hw.fittedvalues[-3:]
                actual    = monthly_ts['revenue'].iloc[-3:]
                mape = float(np.mean(np.abs((actual.values - in_sample.values) / actual.values)) * 100)
            else:
                mape = 0.0

            col_fc1, col_fc2 = st.columns([3, 1])
            with col_fc1:
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(
                    x=monthly_ts.index, y=monthly_ts['revenue'],
                    fill='tozeroy', fillcolor='rgba(0,177,79,0.12)',
                    line=dict(color='#00B14F', width=2.5),
                    mode='lines+markers', name='Aktual',
                    hovertemplate='%{x|%b %Y}<br>R$ %{y:,.0f}<extra></extra>'
                ))
                fig_fc.add_trace(go.Scatter(
                    x=list(forecast_idx) + list(forecast_idx[::-1]),
                    y=list(forecast_vals * 1.10) + list(forecast_vals[::-1] * 0.90),
                    fill='toself', fillcolor='rgba(255,107,53,0.15)',
                    line=dict(color='rgba(255,107,53,0)'), name='CI ±10%'
                ))
                fig_fc.add_trace(go.Scatter(
                    x=forecast_idx, y=forecast_vals,
                    line=dict(color='#FF6B35', width=2.5, dash='dash'),
                    mode='lines+markers+text',
                    text=[f"R${v/1e6:.2f}M" for v in forecast_vals],
                    textposition='top center', name='Forecast',
                    hovertemplate='%{x|%b %Y}<br>Forecast: R$ %{y:,.0f}<extra></extra>'
                ))
                fig_fc.add_vline(x=monthly_ts.index[-1].timestamp()*1000,
                                 line_dash='dot', line_color='gray',
                                 annotation_text=' Batas Forecast')
                fig_fc.update_layout(
                    title=dict(text=f'Revenue Forecast 3 Bulan ke Depan | Holt-Winters | MAPE: {mape:.1f}%',
                               font=dict(size=14, color='#00521f', family='Arial Black')),
                    height=380, plot_bgcolor='white', paper_bgcolor='white',
                    margin=dict(t=45,b=40,l=60,r=20), hovermode='x unified',
                    xaxis=dict(showgrid=False), yaxis=dict(gridcolor='#f0f0f0', tickprefix='R$ ', tickformat=',.0f'),
                    legend=dict(bgcolor='white', bordercolor='#e0e0e0', borderwidth=1)
                )
                st.plotly_chart(fig_fc, use_container_width=True)

            with col_fc2:
                st.markdown("<div class='section-title'>🔮 Hasil Forecast</div>", unsafe_allow_html=True)
                for d, v in zip(forecast_idx, forecast_vals):
                    st.markdown(f"""<div class='kpi-card' style='height:auto;margin-bottom:8px'>
                        <div class='kpi-label'>{d.strftime('%B %Y')}</div>
                        <div class='kpi-value' style='font-size:18px'>R$ {v/1e6:.3f}M</div>
                        <span class='kpi-growth-pos'>Forecast</span></div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class='insight-box'>
                    💡 Model: <b>Holt-Winters</b><br>MAPE: <b>{mape:.1f}%</b><br>
                    {'Sangat Akurat ✅' if mape<10 else 'Cukup Akurat ✅' if mape<20 else 'Perlu Validasi ⚠️'}
                </div>""", unsafe_allow_html=True)

            st.markdown("<div class='section-title'>📋 Detail Forecast</div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                'Bulan': [d.strftime('%B %Y') for d in forecast_idx],
                'Forecast (R$)': [f'{v:,.2f}' for v in forecast_vals],
                'CI Lower (R$)': [f'{v*0.9:,.2f}' for v in forecast_vals],
                'CI Upper (R$)': [f'{v*1.1:,.2f}' for v in forecast_vals],
            }), use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error fitting model: {e}")

# ════════════════════════════════════════════
# TAB 6: ML CHURN PREDICTION
# ════════════════════════════════════════════
with tab6:
    st.markdown("<div class='section-title'>🤖 Machine Learning — Customer Churn Prediction</div>", unsafe_allow_html=True)
    st.markdown("""<div class='insight-box' style='margin-bottom:12px'>
        <b>Definisi Churn:</b> Pelanggan dengan Frequency = 1 (hanya beli sekali) DAN Recency di atas median.
        Model membandingkan Logistic Regression, Random Forest, dan Gradient Boosting.
    </div>""", unsafe_allow_html=True)

    churn_df = rfm_data.copy()
    median_rec = churn_df['Recency'].median()
    churn_df['Churn'] = ((churn_df['Frequency'] == 1) & (churn_df['Recency'] > median_rec)).astype(int)
    features_ml = ['Recency','Frequency','Monetary']
    X_ml = churn_df[features_ml]
    y_ml = churn_df['Churn']

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>👥 Total Pelanggan</div>
            <div class='kpi-value'>{len(churn_df):,}</div></div>""", unsafe_allow_html=True)
    with cc2:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>⚠️ Churn Rate</div>
            <div class='kpi-value' style='color:#FF5252'>{y_ml.mean()*100:.1f}%</div></div>""", unsafe_allow_html=True)
    with cc3:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>✅ Retention Rate</div>
            <div class='kpi-value' style='color:#00B14F'>{(1-y_ml.mean())*100:.1f}%</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.spinner("🤖 Melatih model ML..."):
        X_tr_ml, X_te_ml, y_tr_ml, y_te_ml = train_test_split(X_ml, y_ml, test_size=0.2, random_state=42, stratify=y_ml)
        sc_ml = StandardScaler()
        Xtr_sc = sc_ml.fit_transform(X_tr_ml)
        Xte_sc = sc_ml.transform(X_te_ml)
        ml_results = {}
        for mname, (mobj, Xtr_, Xte_) in {
            'Logistic Regression': (LogisticRegression(max_iter=500, random_state=42), Xtr_sc, Xte_sc),
            'Random Forest':       (RandomForestClassifier(n_estimators=100, random_state=42), X_tr_ml, X_te_ml),
            'Gradient Boosting':   (GradientBoostingClassifier(n_estimators=100, random_state=42), X_tr_ml, X_te_ml),
        }.items():
            mobj.fit(Xtr_, y_tr_ml)
            prob_ = mobj.predict_proba(Xte_)[:,1]
            ml_results[mname] = {'model': mobj, 'prob': prob_, 'auc': roc_auc_score(y_te_ml, prob_)}

    best_ml = max(ml_results, key=lambda k: ml_results[k]['auc'])

    mc1, mc2 = st.columns(2)
    with mc1:
        fig_roc = go.Figure()
        for (nm, rs), clr in zip(ml_results.items(), ['#00B14F','#FF6B35','#1DE9B6']):
            fpr_, tpr_, _ = roc_curve(y_te_ml, rs['prob'])
            fig_roc.add_trace(go.Scatter(x=fpr_, y=tpr_, name=f"{nm} (AUC={rs['auc']:.3f})", line=dict(color=clr, width=2.5)))
        fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],line=dict(color='gray',dash='dash',width=1.5),showlegend=False))
        fig_roc.update_layout(
            title=dict(text='ROC Curve — Perbandingan Model', font=dict(size=13,color='#00521f',family='Arial Black')),
            height=320, plot_bgcolor='white', paper_bgcolor='white', margin=dict(t=40,b=40,l=50,r=20),
            xaxis=dict(title='FPR',showgrid=False), yaxis=dict(title='TPR',gridcolor='#f0f0f0'),
            legend=dict(font=dict(size=10),bgcolor='white',bordercolor='#e0e0e0',borderwidth=1)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with mc2:
        rf_imp = pd.Series(ml_results['Random Forest']['model'].feature_importances_, index=features_ml).sort_values()
        fig_imp = go.Figure(go.Bar(
            x=rf_imp.values, y=rf_imp.index, orientation='h',
            marker=dict(color=['#00521f' if v > rf_imp.median() else '#B2DFDB' for v in rf_imp.values]),
            text=[f'{v:.3f}' for v in rf_imp.values], textposition='outside', textfont=dict(size=11)
        ))
        fig_imp.update_layout(
            title=dict(text='Feature Importance (Random Forest)', font=dict(size=13,color='#00521f',family='Arial Black')),
            height=320, plot_bgcolor='white', paper_bgcolor='white', margin=dict(t=40,b=40,l=100,r=60),
            xaxis=dict(showgrid=False,title='Importance'), yaxis=dict(gridcolor='#f0f0f0')
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("<div class='section-title'>📋 Perbandingan Model</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([
        {'Model': nm, 'ROC-AUC': f"{rs['auc']:.4f}", 'Status': '🏆 Best' if nm==best_ml else ''}
        for nm, rs in ml_results.items()
    ]), use_container_width=True, hide_index=True)
    st.markdown(f"""<div class='insight-box'>
        💡 <b>Best Model: {best_ml}</b> (AUC = {ml_results[best_ml]['auc']:.4f})<br>
        <b>Recency</b> adalah fitur paling prediktif — pelanggan yang lama tidak transaksi sangat berisiko churn.<br>
        <b>Rekomendasi:</b> Kirim win-back campaign otomatis ke pelanggan At Risk.
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════
# TAB 7: ADVANCED VISUALIZATION
# ════════════════════════════════════════════
with tab7:
    st.markdown("<div class='section-title'>📊 Advanced Interactive Visualization</div>", unsafe_allow_html=True)

    inner_tabs = st.tabs(["🌳 Treemap", "🫧 Bubble Chart", "🔥 Heatmap", "🔵 3D RFM"])

    with inner_tabs[0]:
        treemap_df = (
            filtered_df.groupby('product_category_name_english')['payment_value']
            .sum().reset_index().rename(columns={'product_category_name_english':'category','payment_value':'revenue'})
            .nlargest(20,'revenue')
        )
        fig_tm = px.treemap(treemap_df, path=['category'], values='revenue', color='revenue',
                            color_continuous_scale=['#c8f5d8','#00521f'],
                            title='<b>Revenue Breakdown — Top 20 Kategori</b>')
        fig_tm.update_traces(texttemplate='<b>%{label}</b><br>R$%{value:,.0f}<br>%{percentRoot:.1%}')
        fig_tm.update_layout(height=500, paper_bgcolor='white', title_font=dict(size=14,color='#00521f'))
        st.plotly_chart(fig_tm, use_container_width=True)
        st.markdown("<div class='insight-box'>💡 Ukuran kotak = kontribusi revenue. Hover untuk detail nilai & persentase.</div>", unsafe_allow_html=True)

    with inner_tabs[1]:
        bub_df = (
            filtered_df.groupby('product_category_name_english').agg(
                revenue=('payment_value','sum'), orders=('order_id','nunique'), avg_val=('payment_value','mean')
            ).reset_index().query('orders >= 30').nlargest(25,'revenue')
        )
        fig_bub = px.scatter(bub_df, x='orders', y='revenue', size='avg_val', color='avg_val',
                             hover_name='product_category_name_english',
                             color_continuous_scale='RdYlGn', size_max=45,
                             title='<b>Revenue vs Orders vs Avg Transaction Value</b>',
                             labels={'orders':'Jumlah Orders','revenue':'Total Revenue','avg_val':'Avg Transaction'})
        fig_bub.update_layout(height=480, plot_bgcolor='white', paper_bgcolor='white',
                              title_font=dict(size=14,color='#00521f'))
        st.plotly_chart(fig_bub, use_container_width=True)
        st.markdown("<div class='insight-box'>💡 Ukuran gelembung = nilai transaksi rata-rata. Pojok kanan atas = high volume & high revenue.</div>", unsafe_allow_html=True)

    with inner_tabs[2]:
        hm_df = filtered_df.copy()
        hm_df['dow'] = hm_df['order_purchase_timestamp'].dt.day_name()
        hm_df['mon'] = hm_df['order_purchase_timestamp'].dt.month
        day_ord = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        pivot_hm = (hm_df.groupby(['mon','dow'])['payment_value'].sum().reset_index()
                    .pivot(index='dow',columns='mon',values='payment_value').reindex(day_ord))
        mon_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        pivot_hm.columns = [mon_labels[c-1] for c in pivot_hm.columns]
        fig_hm = px.imshow(pivot_hm, color_continuous_scale=['#f0fdf4','#00521f'],
                           labels=dict(x='Bulan',y='Hari',color='Revenue'),
                           title='<b>Heatmap — Pola Revenue per Hari & Bulan</b>',
                           text_auto='.2s', aspect='auto')
        fig_hm.update_layout(height=400, paper_bgcolor='white', title_font=dict(size=14,color='#00521f'))
        st.plotly_chart(fig_hm, use_container_width=True)
        st.markdown("<div class='insight-box'>💡 Warna gelap = revenue tertinggi. Gunakan untuk menjadwalkan promo & iklan pada hari & bulan puncak.</div>", unsafe_allow_html=True)

    with inner_tabs[3]:
        rfm_s = rfm_data.sample(min(3000, len(rfm_data)), random_state=42)
        fig_3d = px.scatter_3d(rfm_s, x='Recency', y='Frequency', z='Monetary', color='Segment',
                               color_discrete_map=segment_colors, opacity=0.7,
                               title='<b>3D RFM — Distribusi Pelanggan</b>',
                               labels={'Recency':'Recency (hari)','Frequency':'Frekuensi','Monetary':'Nilai Belanja'})
        fig_3d.update_layout(height=580, paper_bgcolor='white',
                             title_font=dict(size=14,color='#00521f'),
                             scene=dict(xaxis=dict(backgroundcolor='#f0fdf4'),
                                        yaxis=dict(backgroundcolor='#f0fdf4'),
                                        zaxis=dict(backgroundcolor='#f0fdf4')))
        st.plotly_chart(fig_3d, use_container_width=True)
        st.markdown("<div class='insight-box'>💡 Rotasi grafik dengan drag. Champions terkonsentrasi di Recency rendah + Frequency & Monetary tinggi.</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# TAB 8: A/B TESTING
# ════════════════════════════════════════════
with tab8:
    st.markdown("<div class='section-title'>🧪 A/B Testing — Uji Statistik Metode Pembayaran</div>", unsafe_allow_html=True)
    st.markdown("""<div class='insight-box' style='margin-bottom:12px'>
        <b>Hipotesis:</b> H₀: Tidak ada perbedaan nilai transaksi antar metode pembayaran |
        H₁: Ada perbedaan signifikan (α = 0.05). Menggunakan Mann-Whitney U (non-parametric).
    </div>""", unsafe_allow_html=True)

    ab_col1, ab_col2 = st.columns(2)
    payment_types_ab = filtered_df['payment_type'].dropna().unique().tolist()
    with ab_col1:
        ga_type = st.selectbox("Group A (Treatment)", payment_types_ab,
                                index=payment_types_ab.index('credit_card') if 'credit_card' in payment_types_ab else 0)
    with ab_col2:
        rem_types = [p for p in payment_types_ab if p != ga_type]
        gb_type = st.selectbox("Group B (Control)", rem_types,
                                index=rem_types.index('boleto') if 'boleto' in rem_types else 0)

    gA = filtered_df[filtered_df['payment_type']==ga_type]['payment_value'].dropna()
    gB = filtered_df[filtered_df['payment_type']==gb_type]['payment_value'].dropna()

    if len(gA) < 30 or len(gB) < 30:
        st.warning("Data terlalu sedikit. Perluas filter untuk A/B test.")
    else:
        n_ab = min(5000, len(gA), len(gB))
        np.random.seed(42)
        sA = gA.sample(n_ab); sB = gB.sample(n_ab)
        _, p_mw = stats.mannwhitneyu(sA, sB, alternative='two-sided')
        pooled_s = np.sqrt((sA.std()**2 + sB.std()**2)/2)
        cd = (sA.mean() - sB.mean()) / pooled_s if pooled_s > 0 else 0
        eff_lbl = 'Kecil' if abs(cd)<0.2 else ('Sedang' if abs(cd)<0.8 else 'Besar')
        is_sig = p_mw < 0.05

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""<div class='kpi-card'><div class='kpi-label'>📊 Mean Group A</div>
                <div class='kpi-value' style='font-size:18px'>R$ {sA.mean():,.0f}</div>
                <span class='kpi-growth-pos'>{ga_type.replace('_',' ').title()}</span></div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""<div class='kpi-card'><div class='kpi-label'>📊 Mean Group B</div>
                <div class='kpi-value' style='font-size:18px'>R$ {sB.mean():,.0f}</div>
                <span class='kpi-growth-pos'>{gb_type.replace('_',' ').title()}</span></div>""", unsafe_allow_html=True)
        with k3:
            p_clr = "#00B14F" if is_sig else "#FF5252"
            st.markdown(f"""<div class='kpi-card'><div class='kpi-label'>🧮 p-value (Mann-Whitney)</div>
                <div class='kpi-value' style='font-size:18px;color:{p_clr}'>{p_mw:.4f}</div>
                <span style='color:{p_clr};font-size:11px;font-weight:700'>{'✅ Signifikan' if is_sig else '❌ Tidak Signifikan'}</span></div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""<div class='kpi-card'><div class='kpi-label'>📐 Cohen's d</div>
                <div class='kpi-value' style='font-size:18px'>{cd:.3f}</div>
                <span class='kpi-growth-pos'>{eff_lbl}</span></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        v1, v2 = st.columns(2)
        cap_ab = max(sA.quantile(0.95), sB.quantile(0.95))

        with v1:
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=sA[sA<=cap_ab], nbinsx=40, name=f'Group A ({ga_type})',
                                            marker_color='#00B14F', opacity=0.65, histnorm='probability density'))
            fig_dist.add_trace(go.Histogram(x=sB[sB<=cap_ab], nbinsx=40, name=f'Group B ({gb_type})',
                                            marker_color='#FF6B35', opacity=0.65, histnorm='probability density'))
            fig_dist.add_vline(x=sA.mean(), line_color='#00521f', line_dash='dash', line_width=2)
            fig_dist.add_vline(x=sB.mean(), line_color='#C84B11', line_dash='dash', line_width=2)
            fig_dist.update_layout(
                title=dict(text='Distribusi Nilai Transaksi', font=dict(size=13,color='#00521f',family='Arial Black')),
                height=320, plot_bgcolor='white', paper_bgcolor='white', barmode='overlay',
                margin=dict(t=40,b=40,l=50,r=20),
                xaxis=dict(showgrid=False,title='Nilai Transaksi (BRL)'),
                yaxis=dict(gridcolor='#f0f0f0',title='Density'),
                legend=dict(bgcolor='white',bordercolor='#e0e0e0',borderwidth=1)
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with v2:
            sig_text = f"{'***' if p_mw<0.001 else '**' if p_mw<0.01 else '*' if p_mw<0.05 else 'ns'} (p={p_mw:.4f})"
            fig_box_ab = go.Figure()
            fig_box_ab.add_trace(go.Box(y=sA[sA<=cap_ab], name=f'Group A\n{ga_type}',
                                        marker_color='#00B14F', boxmean='sd', notched=True))
            fig_box_ab.add_trace(go.Box(y=sB[sB<=cap_ab], name=f'Group B\n{gb_type}',
                                        marker_color='#FF6B35', boxmean='sd', notched=True))
            fig_box_ab.update_layout(
                title=dict(text=f'Box Plot — {sig_text}', font=dict(size=13,color='#00521f',family='Arial Black')),
                height=320, plot_bgcolor='white', paper_bgcolor='white',
                margin=dict(t=40,b=40,l=50,r=20),
                yaxis=dict(title='Nilai Transaksi (BRL)',gridcolor='#f0f0f0'),
                legend=dict(bgcolor='white',bordercolor='#e0e0e0',borderwidth=1)
            )
            st.plotly_chart(fig_box_ab, use_container_width=True)

        if is_sig:
            winner = ga_type if sA.mean() > sB.mean() else gb_type
            st.markdown(f"""<div class='insight-box'>
                ✅ <b>H₁ DITERIMA</b> — Ada perbedaan signifikan (p={p_mw:.4f} &lt; 0.05)<br>
                <b>{winner.replace('_',' ').title()}</b> menghasilkan nilai transaksi lebih tinggi rata-rata
                <b>R$ {abs(sA.mean()-sB.mean()):,.2f}</b> per transaksi. Effect size: <b>{eff_lbl}</b> (d={cd:.3f})<br>
                💡 <b>Rekomendasi:</b> Dorong penggunaan {winner.replace('_',' ')} melalui cashback atau cicilan 0%.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='insight-box' style='border-color:#FFB300;background:linear-gradient(135deg,#fffbeb,#fef3c7)'>
                ❌ <b>H₀ GAGAL DITOLAK</b> — Tidak ada perbedaan signifikan (p={p_mw:.4f} ≥ 0.05)<br>
                💡 Coba perluas rentang data atau bandingkan segmen pelanggan yang berbeda.
            </div>""", unsafe_allow_html=True)

# ─── FOOTER ────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#aaa; font-size:12px; padding:8px'>
    🛒 E-Commerce Analytics Dashboard | Olist Brazil 2016-2018 | Built with Streamlit
</div>""", unsafe_allow_html=True)