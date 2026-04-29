import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import os
from st_aggrid import AgGrid, GridOptionsBuilder

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
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Overview", "🛍️ Product Analysis", "👥 Customer", "🗺️ Geospatial"
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

# ─── FOOTER ────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#aaa; font-size:12px; padding:8px'>
    🛒 E-Commerce Analytics Dashboard | Olist Brazil 2016-2018 | Built with Streamlit
</div>""", unsafe_allow_html=True)