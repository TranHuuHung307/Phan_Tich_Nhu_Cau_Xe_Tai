#!/usr/bin/env python
# coding: utf-8

# # 📦 PHÂN TÍCH NHU CẦU XE TẢI MÙA VỤ - 5 NĂM
# ### Lĩnh vực: Vận tải / Logistics tại Việt Nam
# 
# **Mục tiêu phân tích:**
# - Xử lý và làm sạch dữ liệu chuỗi thời gian về nhu cầu xe tải trên 2 tuyến đường HN-HCM và HCM-HN trong vòng 5 năm.
# - Trực quan hóa xu hướng vận tải theo thời gian, theo tuyến đường và theo loại phương tiện.
# - Phân rã chuỗi thời gian để xác định xu hướng (Trend), tính mùa vụ (Seasonality) và nhiễu (Residual).
# 
# **Nguồn dữ liệu:** `Du_Bao_Nhu_Cau_Xe_Tai_Mua_Vu_5Nam.xlsx`
# 
# ---

# ## ⚙️ CELL 1: KHAI BÁO THƯ VIỆN

# In[1]:


# ============================================================
# CELL 1: Khai báo toàn bộ thư viện cần thiết cho dự án
# ============================================================

# Thư viện xử lý dữ liệu
import pandas as pd
import numpy as np

# Thư viện vẽ biểu đồ
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Thư viện phân tích chuỗi thời gian
from statsmodels.tsa.seasonal import seasonal_decompose

# Cấu hình hiển thị biểu đồ
plt.rcParams['figure.figsize'] = (14, 6)       # Kích thước mặc định cho biểu đồ
plt.rcParams['axes.titlesize'] = 16            # Cỡ chữ tiêu đề biểu đồ
plt.rcParams['axes.labelsize'] = 13            # Cỡ chữ tên trục
plt.rcParams['xtick.labelsize'] = 11           # Cỡ chữ nhãn trục X
plt.rcParams['ytick.labelsize'] = 11           # Cỡ chữ nhãn trục Y
plt.rcParams['legend.fontsize'] = 11           # Cỡ chữ chú thích
plt.rcParams['figure.dpi'] = 120               # Độ phân giải biểu đồ (ảnh sắc nét hơn)

# Thiết lập style cho seaborn
sns.set_theme(style='whitegrid', palette='muted')

# Tắt cảnh báo không cần thiết
import warnings
warnings.filterwarnings('ignore')

print('✅ Khai báo thư viện thành công!')


# ## 📂 CELL 2: ĐỌC DỮ LIỆU VÀ KIỂM TRA SƠ BỘ

# In[2]:


# ============================================================
# CELL 2: Đọc file dữ liệu từ cùng thư mục chứa file notebook
# ============================================================

# Đọc file Excel (sử dụng tên file tương đối - chỉ cần đặt file Excel
# cùng thư mục với file Notebook này là chạy được ngay)
df = pd.read_excel('Du_Bao_Nhu_Cau_Xe_Tai_Mua_Vu_5Nam.xlsx')

print('=' * 60)
print('📋 THÔNG TIN TỔNG QUAN VỀ DỮ LIỆU')
print('=' * 60)
print(f'  • Số dòng (quan sát): {df.shape[0]:,}')
print(f'  • Số cột (biến):      {df.shape[1]}')
print(f'  • Các cột có trong bộ dữ liệu: {list(df.columns)}')
print()

print('📌 Kiểu dữ liệu của từng cột:')
print(df.dtypes)
print()

print('📌 5 dòng dữ liệu đầu tiên:')
display(df.head())

print('📌 5 dòng dữ liệu cuối cùng:')
display(df.tail())

print('📌 Thống kê mô tả (Descriptive Statistics):')
display(df.describe())


# ---
# ## 🧹 NHIỆM VỤ 1: XỬ LÝ DỮ LIỆU (DATA PREPROCESSING)

# ### CELL 3: Kiểm tra và xử lý giá trị khuyết (Missing Values)

# In[3]:


# ============================================================
# CELL 3: Kiểm tra và xử lý giá trị khuyết (NaN / Null)
# ============================================================

print('=' * 60)
print('🔍 BƯỚC 1: KIỂM TRA GIÁ TRỊ KHUYẾT (MISSING VALUES)')
print('=' * 60)

# Đếm số lượng giá trị khuyết trong từng cột
missing_count = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df) * 100).round(2)

missing_report = pd.DataFrame({
    'Số lượng giá trị khuyết': missing_count,
    'Tỷ lệ (%)': missing_percent
})

print(missing_report)
print()

# Kiểm tra tổng số giá trị khuyết
total_missing = df.isnull().sum().sum()

if total_missing == 0:
    print('✅ Bộ dữ liệu KHÔNG có giá trị khuyết. Không cần xử lý thêm bước này.')
else:
    print(f'⚠️  Phát hiện tổng cộng {total_missing} giá trị khuyết. Tiến hành xử lý...')

    # --- Phương pháp xử lý phù hợp cho chuỗi thời gian ---
    # Sắp xếp dữ liệu theo thời gian trước khi điền giá trị khuyết
    df = df.sort_values(by=['Year', 'Week', 'Route', 'Fleet_Type']).reset_index(drop=True)

    # Bước 1: Dùng phương pháp 'forward fill' (ffill) - điền bằng giá trị liền trước
    # Phù hợp nhất cho chuỗi thời gian vì giả định giá trị ít thay đổi đột ngột
    df['Actual_Trips'] = df['Actual_Trips'].ffill()
    df['Total_Volume_Tons'] = df['Total_Volume_Tons'].ffill()

    # Bước 2: Nếu vẫn còn (đầu chuỗi), dùng 'backward fill' (bfill)
    df['Actual_Trips'] = df['Actual_Trips'].bfill()
    df['Total_Volume_Tons'] = df['Total_Volume_Tons'].bfill()

    print(f'✅ Đã xử lý xong giá trị khuyết bằng phương pháp Forward Fill + Backward Fill.')
    print(f'   Kiểm tra lại: Tổng giá trị khuyết còn lại = {df.isnull().sum().sum()}')


# ### CELL 4: Kiểm tra và xử lý giá trị ngoại lai (Outliers)

# In[4]:


# ============================================================
# CELL 4: Phát hiện ngoại lai bằng phương pháp IQR
# Lưu ý: CHỈ xét các tuần THƯỜNG (Is_Peak_Event = 0),
# loại trừ các tuần cao điểm lễ tết vì chúng có giá trị
# cao bất thường là do bản chất mùa vụ, không phải ngoại lai.
# ============================================================

print('=' * 60)
print('🔍 BƯỚC 2: PHÁT HIỆN NGOẠI LAI (OUTLIERS) - PHƯƠNG PHÁP IQR')
print('=' * 60)
print('⚠️  Chỉ áp dụng trên dữ liệu tuần THƯỜNG (Is_Peak_Event = 0)')
print('   (Các tuần lễ tết bị loại trừ vì giá trị cao là do mùa vụ, không phải lỗi dữ liệu)')
print()

# Lọc ra chỉ các tuần thường (không phải cao điểm lễ tết)
df_normal = df[df['Is_Peak_Event'] == 0].copy()

def detect_outliers_iqr(data, column):
    """Hàm phát hiện ngoại lai bằng phương pháp IQR (Interquartile Range)"""
    Q1 = data[column].quantile(0.25)  # Tứ phân vị thứ 1
    Q3 = data[column].quantile(0.75)  # Tứ phân vị thứ 3
    IQR = Q3 - Q1                      # Khoảng tứ phân vị

    # Ngưỡng phát hiện ngoại lai: nằm ngoài khoảng [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Đánh dấu các dòng là ngoại lai
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]

    print(f'  📊 Cột [{column}]:')
    print(f'     Q1 = {Q1:.2f} | Q3 = {Q3:.2f} | IQR = {IQR:.2f}')
    print(f'     Ngưỡng dưới (Lower Bound) = {lower_bound:.2f}')
    print(f'     Ngưỡng trên (Upper Bound) = {upper_bound:.2f}')
    print(f'     Số lượng ngoại lai phát hiện = {len(outliers)} dòng ({len(outliers)/len(data)*100:.2f}%)')
    print()
    return lower_bound, upper_bound

# Phát hiện ngoại lai cho cột Actual_Trips
lb_trips, ub_trips = detect_outliers_iqr(df_normal, 'Actual_Trips')

# Phát hiện ngoại lai cho cột Total_Volume_Tons
lb_vol, ub_vol = detect_outliers_iqr(df_normal, 'Total_Volume_Tons')

# --- Trực quan hóa ngoại lai bằng Boxplot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot cho Actual_Trips
axes[0].boxplot(df_normal['Actual_Trips'], patch_artist=True,
                boxprops=dict(facecolor='#4ECDC4', color='#2c3e50'),
                medianprops=dict(color='#e74c3c', linewidth=2))
axes[0].set_title('Boxplot: Actual_Trips\n(Tuần thường - không phải lễ tết)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Số chuyến xe (chuyến)')
axes[0].set_xticks([])

# Boxplot cho Total_Volume_Tons
axes[1].boxplot(df_normal['Total_Volume_Tons'], patch_artist=True,
                boxprops=dict(facecolor='#F7DC6F', color='#2c3e50'),
                medianprops=dict(color='#e74c3c', linewidth=2))
axes[1].set_title('Boxplot: Total_Volume_Tons\n(Tuần thường - không phải lễ tết)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Khối lượng hàng hóa (Tấn)')
axes[1].set_xticks([])

plt.suptitle('Kiểm Tra Ngoại Lai (Outlier Detection) - Phương Pháp IQR', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('01_Outlier_Boxplot.png', bbox_inches='tight', dpi=150)
plt.show()
print('✅ Đã lưu biểu đồ: 01_Outlier_Boxplot.png')


# ### CELL 5: Gộp dữ liệu từ mức Tuần lên mức Tháng (Resampling)

# In[5]:


# ============================================================
# CELL 5: Gộp (Aggregate) dữ liệu từ mức TUẦN lên mức THÁNG
# Mục đích: Giảm nhiễu ngắn hạn, làm nổi bật xu hướng dài hạn
# và chuẩn bị cho bước phân rã chuỗi thời gian (Decomposition)
# ============================================================

print('=' * 60)
print('🔍 BƯỚC 3: GỘP DỮ LIỆU TUẦN -> THÁNG (RESAMPLING)')
print('=' * 60)

# Tạo cột 'Date' là ngày đầu tiên của mỗi tuần trong năm
# Dùng ISO week date: thứ Hai đầu tuần để xác định ngày
df['Date'] = pd.to_datetime(
    df['Year'].astype(str) + '-W' + df['Week'].astype(str).str.zfill(2) + '-1',
    format='%G-W%V-%u'
)

# Tạo cột 'YearMonth' để gộp theo tháng
df['YearMonth'] = df['Date'].dt.to_period('M')

# --- Gộp toàn bộ dữ liệu theo Tháng (tổng hợp tất cả tuyến và loại xe) ---
df_monthly = df.groupby('YearMonth').agg(
    Total_Trips=('Actual_Trips', 'sum'),
    Total_Volume=('Total_Volume_Tons', 'sum'),
    Peak_Weeks=('Is_Peak_Event', 'sum')   # Đếm số tuần cao điểm trong tháng
).reset_index()

# Chuyển YearMonth sang kiểu datetime để vẽ biểu đồ
df_monthly['Date'] = df_monthly['YearMonth'].dt.to_timestamp()

# --- Gộp theo Tháng và Tuyến đường (để so sánh tuyến HN-HCM vs HCM-HN) ---
df_monthly_route = df.groupby(['YearMonth', 'Route']).agg(
    Total_Trips=('Actual_Trips', 'sum'),
    Total_Volume=('Total_Volume_Tons', 'sum')
).reset_index()
df_monthly_route['Date'] = df_monthly_route['YearMonth'].dt.to_timestamp()

# --- Gộp theo Năm và Tháng trong năm (để vẽ bar chart so sánh theo tháng) ---
df['Month'] = df['Date'].dt.month
df_by_month = df.groupby(['Year', 'Month']).agg(
    Total_Trips=('Actual_Trips', 'sum'),
    Total_Volume=('Total_Volume_Tons', 'sum')
).reset_index()

# --- Gộp theo Năm và Tuyến đường ---
df_by_route = df.groupby('Route').agg(
    Total_Trips=('Actual_Trips', 'sum'),
    Total_Volume=('Total_Volume_Tons', 'sum')
).reset_index()

# --- Gộp theo Loại xe ---
df_by_fleet = df.groupby('Fleet_Type').agg(
    Total_Volume=('Total_Volume_Tons', 'sum')
).reset_index()

print(f'✅ Gộp dữ liệu theo tháng hoàn tất!')
print(f'   Số tháng trong bộ dữ liệu gộp: {len(df_monthly)} tháng')
print()
print('📌 Mẫu dữ liệu theo tháng (5 dòng đầu):')
display(df_monthly.head())


# ---
# ## 📊 NHIỆM VỤ 2: TRỰC QUAN HÓA DỮ LIỆU

# ### CELL 6: Biểu đồ 1 - Line Chart (Xu hướng theo thời gian)

# In[6]:


# ============================================================
# CELL 6: BIỂU ĐỒ 1 - LINE CHART
# Thể hiện biến động của Tổng khối lượng hàng hóa (Total_Volume_Tons)
# và Số chuyến xe (Actual_Trips) theo thời gian qua 5 năm.
# Mục tiêu: Nhìn thấy xu hướng tổng thể và các đỉnh cao điểm mùa vụ.
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# --- Subplot 1: Tổng khối lượng hàng hóa theo tháng ---
axes[0].plot(df_monthly['Date'], df_monthly['Total_Volume'],
             color='#2E86AB', linewidth=2.5, marker='o', markersize=4, label='Tổng KL hàng hóa')

# Tô màu vùng có tuần cao điểm (lễ tết)
peak_months = df_monthly[df_monthly['Peak_Weeks'] > 0]
for _, row in peak_months.iterrows():
    axes[0].axvspan(row['Date'] - pd.Timedelta(days=15),
                    row['Date'] + pd.Timedelta(days=15),
                    alpha=0.2, color='#E74C3C', label='_nolegend_')

# Thêm đường bình quân động (Moving Average 3 tháng)
df_monthly['MA3_Volume'] = df_monthly['Total_Volume'].rolling(window=3, center=True).mean()
axes[0].plot(df_monthly['Date'], df_monthly['MA3_Volume'],
             color='#E74C3C', linewidth=2, linestyle='--', label='Trung bình động 3 tháng')

axes[0].set_title('Biến Động Tổng Khối Lượng Hàng Hóa Vận Chuyển Theo Tháng (2021-2025)',
                  fontsize=15, fontweight='bold', pad=12)
axes[0].set_xlabel('Thời gian (Tháng/Năm)')
axes[0].set_ylabel('Tổng Khối Lượng (Tấn)')
axes[0].legend(loc='upper left')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# Thêm chú thích vùng tô màu đỏ (cao điểm)
from matplotlib.patches import Patch
legend_patch = Patch(color='#E74C3C', alpha=0.2, label='Tuần cao điểm lễ tết')
axes[0].legend(handles=axes[0].get_legend_handles_labels()[0] + [legend_patch],
               labels=axes[0].get_legend_handles_labels()[1] + ['Tuần cao điểm lễ tết'],
               loc='upper left')

# --- Subplot 2: Tổng số chuyến xe theo tháng ---
axes[1].plot(df_monthly['Date'], df_monthly['Total_Trips'],
             color='#A23B72', linewidth=2.5, marker='s', markersize=4, label='Tổng số chuyến xe')

# Thêm đường bình quân động (Moving Average 3 tháng)
df_monthly['MA3_Trips'] = df_monthly['Total_Trips'].rolling(window=3, center=True).mean()
axes[1].plot(df_monthly['Date'], df_monthly['MA3_Trips'],
             color='#F18F01', linewidth=2, linestyle='--', label='Trung bình động 3 tháng')

for _, row in peak_months.iterrows():
    axes[1].axvspan(row['Date'] - pd.Timedelta(days=15),
                    row['Date'] + pd.Timedelta(days=15),
                    alpha=0.2, color='#E74C3C', label='_nolegend_')

axes[1].set_title('Biến Động Tổng Số Chuyến Xe Vận Chuyển Theo Tháng (2021-2025)',
                  fontsize=15, fontweight='bold', pad=12)
axes[1].set_xlabel('Thời gian (Tháng/Năm)')
axes[1].set_ylabel('Tổng Số Chuyến Xe (chuyến)')
axes[1].legend(loc='upper left')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

plt.tight_layout(pad=3.0)
plt.savefig('02_LineChart_XuHuong.png', bbox_inches='tight', dpi=150)
plt.show()
print('✅ Đã lưu biểu đồ: 02_LineChart_XuHuong.png')


# ### 📝 NHẬN XÉT & INSIGHT - BIỂU ĐỒ 1 (LINE CHART)
# 
# > **Xu hướng tổng thể (Trend):** Quan sát biểu đồ đường trong giai đoạn 2021–2025, cả tổng khối lượng hàng hóa (`Total_Volume_Tons`) lẫn tổng số chuyến xe (`Actual_Trips`) đều cho thấy xu hướng tăng trưởng bền vững qua từng năm. Điều này phản ánh sự phục hồi và tăng tốc của chuỗi cung ứng hàng hóa Bắc–Nam sau giai đoạn đại dịch, phù hợp với đà tăng trưởng xuất nhập khẩu và thương mại nội địa của Việt Nam.
# >
# > **Tính mùa vụ (Seasonality):** Biểu đồ thể hiện rõ ràng các đỉnh nhu cầu định kỳ, đặc biệt tập trung vào **quý I hàng năm (tháng 1–2)** — trùng với thời điểm trước và sau Tết Nguyên Đán. Đây là giai đoạn các doanh nghiệp logistics ghi nhận lượng hàng hóa tăng đột biến do nhu cầu tích trữ hàng tiêu dùng, thực phẩm và hàng điện tử trước Tết. Sau kỳ nghỉ Tết, nhu cầu hạ nhiệt nhanh trong tháng 2–3 trước khi phục hồi trở lại.
# >
# > **Đường trung bình động 3 tháng (MA3):** Đường MA3 màu đỏ đứt nét giúp làm mượt các biến động ngắn hạn, làm nổi bật xu hướng nền tảng. Khoảng cách giữa đường thực tế và đường MA3 càng lớn trong các tháng cao điểm, chứng tỏ tính mùa vụ có biên độ dao động mạnh và có thể dự báo được với mức độ tin cậy cao.

# ### CELL 7: Biểu đồ 2 - Bar Chart (So sánh theo Tuyến đường / Tháng)

# In[7]:


# ============================================================
# CELL 7: BIỂU ĐỒ 2 - BAR CHART
# So sánh tổng số chuyến xe (Actual_Trips) giữa:
#   - 2 tuyến đường (HN-HCM vs HCM-HN)
#   - Và giữa 12 tháng trong năm (heatmap theo tháng)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Subplot 1: Bar Chart so sánh 2 tuyến đường ---
colors_route = ['#3498DB', '#E74C3C']
bars = axes[0].bar(df_by_route['Route'], df_by_route['Total_Trips'],
                   color=colors_route, edgecolor='white', linewidth=1.5, width=0.5)

# Thêm nhãn giá trị trên đầu mỗi cột
for bar in bars:
    height = bar.get_height()
    axes[0].annotate(f'{height:,.0f}\nchuyến',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 8), textcoords='offset points',
                     ha='center', va='bottom', fontsize=12, fontweight='bold')

axes[0].set_title('Tổng Số Chuyến Xe Theo Tuyến Đường\n(Tích lũy 5 năm 2021–2025)',
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel('Tuyến đường')
axes[0].set_ylabel('Tổng số chuyến xe (chuyến)')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
axes[0].set_ylim(0, df_by_route['Total_Trips'].max() * 1.2)

# --- Subplot 2: Bar Chart theo tháng trong năm (trung bình qua 5 năm) ---
# Tính trung bình số chuyến theo từng tháng trong năm (qua 5 năm)
df_avg_by_month = df_by_month.groupby('Month')['Total_Trips'].mean().reset_index()
month_labels = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6',
                'T7', 'T8', 'T9', 'T10', 'T11', 'T12']

# Tô màu đặc biệt cho tháng có cao điểm lễ tết (tháng 1 và 2)
bar_colors = ['#E74C3C' if m in [1, 2] else '#27AE60' for m in df_avg_by_month['Month']]

bars2 = axes[1].bar(range(1, 13), df_avg_by_month['Total_Trips'],
                    color=bar_colors, edgecolor='white', linewidth=1.2)

# Thêm nhãn giá trị trên đầu mỗi cột
for bar in bars2:
    height = bar.get_height()
    axes[1].annotate(f'{height:,.0f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5), textcoords='offset points',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

axes[1].set_title('Trung Bình Số Chuyến Xe Theo Từng Tháng\n(Trung bình qua 5 năm 2021–2025)',
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel('Tháng trong năm')
axes[1].set_ylabel('Trung bình số chuyến xe (chuyến)')
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels(month_labels)
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# Chú thích màu sắc
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#E74C3C', label='Tháng cao điểm lễ tết'),
                   Patch(facecolor='#27AE60', label='Tháng thường')]
axes[1].legend(handles=legend_elements, loc='upper right')

plt.tight_layout(pad=3.0)
plt.savefig('03_BarChart_SoSanh.png', bbox_inches='tight', dpi=150)
plt.show()
print('✅ Đã lưu biểu đồ: 03_BarChart_SoSanh.png')


# ### 📝 NHẬN XÉT & INSIGHT - BIỂU ĐỒ 2 (BAR CHART)
# 
# > **So sánh theo tuyến đường:** Biểu đồ cột (trái) cho thấy tổng số chuyến vận chuyển tích lũy trong 5 năm giữa hai tuyến HN–HCM và HCM–HN là **tương đối cân bằng**, không có sự chênh lệch quá lớn. Điều này cho thấy dòng chảy hàng hóa trên trục Bắc–Nam là hai chiều khá đều nhau: hàng hóa công nghiệp, điện tử từ miền Bắc xuôi Nam và hàng nông sản, thủy sản từ miền Nam ra Bắc. Doanh nghiệp vận tải có thể tận dụng chiều về để tối ưu hóa chi phí vận hành (tránh xe chạy không tải).
# >
# > **So sánh theo tháng trong năm:** Biểu đồ cột (phải) bộc lộ rõ nét hai đỉnh cao điểm được tô màu đỏ tập trung vào **tháng 1 và tháng 2** — đây là giai đoạn cận Tết và sau Tết Nguyên Đán. Số chuyến xe trong giai đoạn này vượt đáng kể so với mức trung bình của các tháng còn lại trong năm. Đặc biệt, **tháng 2 thường thấp hơn tháng 1** do nghỉ lễ kéo dài. Điều này hàm ý rằng doanh nghiệp logistics cần **lên kế hoạch tăng cường năng lực đội xe và nhân sự từ tháng 11–12** để sẵn sàng đáp ứng nhu cầu đỉnh điểm.

# ### CELL 8: Biểu đồ 3 - Pie Chart (Tỷ trọng theo Loại xe)

# In[8]:


# ============================================================
# CELL 8: BIỂU ĐỒ 3 - PIE CHART
# Thể hiện tỷ trọng phần trăm tổng khối lượng hàng hóa
# được vận chuyển bởi từng loại xe (Truck_15T vs Container_30T)
# Mục tiêu: Xác định loại phương tiện chiếm tỷ trọng chủ đạo.
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Màu sắc cho 2 loại xe
colors_fleet = ['#3498DB', '#E67E22']
explode = (0.05, 0.05)  # Tách nhẹ các mảnh ra để nhìn rõ hơn

# --- Subplot 1: Pie Chart theo Tổng Khối Lượng ---
wedges, texts, autotexts = axes[0].pie(
    df_by_fleet['Total_Volume'],
    labels=df_by_fleet['Fleet_Type'],
    autopct='%1.1f%%',
    colors=colors_fleet,
    explode=explode,
    startangle=90,
    pctdistance=0.75,
    wedgeprops=dict(edgecolor='white', linewidth=2)
)

# Tùy chỉnh style chữ
for text in texts:
    text.set_fontsize(13)
    text.set_fontweight('bold')
for autotext in autotexts:
    autotext.set_fontsize(13)
    autotext.set_fontweight('bold')
    autotext.set_color('white')

# Thêm vòng tròn ở giữa để tạo hình donut chart (đẹp hơn)
centre_circle = plt.Circle((0, 0), 0.50, fc='white')
axes[0].add_patch(centre_circle)

# Thêm giá trị tuyệt đối vào giữa
total_vol = df_by_fleet['Total_Volume'].sum()
axes[0].text(0, 0, f'Tổng\n{total_vol:,.0f}\nTấn',
             ha='center', va='center', fontsize=11, fontweight='bold', color='#2c3e50')

axes[0].set_title('Tỷ Trọng Khối Lượng Hàng Hóa\nTheo Loại Xe (2021–2025)',
                  fontsize=14, fontweight='bold', pad=15)

# --- Subplot 2: Bar Chart ngang bổ sung - So sánh giá trị tuyệt đối ---
fleet_labels = df_by_fleet['Fleet_Type'].tolist()
fleet_values = df_by_fleet['Total_Volume'].tolist()

bars_h = axes[1].barh(fleet_labels, fleet_values,
                      color=colors_fleet, edgecolor='white', linewidth=1.5, height=0.4)

for bar in bars_h:
    width = bar.get_width()
    axes[1].annotate(f'{width:,.0f} Tấn',
                     xy=(width, bar.get_y() + bar.get_height() / 2),
                     xytext=(8, 0), textcoords='offset points',
                     ha='left', va='center', fontsize=12, fontweight='bold')

axes[1].set_title('Khối Lượng Hàng Hóa Tuyệt Đối\nTheo Loại Xe (2021–2025)',
                  fontsize=14, fontweight='bold', pad=15)
axes[1].set_xlabel('Tổng khối lượng (Tấn)')
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
axes[1].set_xlim(0, max(fleet_values) * 1.3)

plt.tight_layout(pad=3.0)
plt.savefig('04_PieChart_LoaiXe.png', bbox_inches='tight', dpi=150)
plt.show()
print('✅ Đã lưu biểu đồ: 04_PieChart_LoaiXe.png')


# ### 📝 NHẬN XÉT & INSIGHT - BIỂU ĐỒ 3 (PIE CHART)
# 
# > **Cơ cấu đội xe:** Biểu đồ tròn thể hiện rõ sự **phân bổ tỷ trọng khối lượng hàng hóa** giữa hai loại phương tiện vận chuyển chính. **Container 30T** chiếm tỷ trọng lớn hơn về tổng khối lượng hàng hóa so với **Truck 15T**, điều này hoàn toàn hợp lý vì container có tải trọng gấp đôi xe tải thông thường, phù hợp với các lô hàng lớn, hàng công nghiệp và hàng xuất nhập khẩu.
# >
# > **Hàm ý chiến lược:** Mặc dù Container 30T vận chuyển khối lượng lớn hơn, Truck 15T lại có ưu thế về **tính linh hoạt** — phù hợp với các tuyến đường nhỏ hơn, giao hàng lẻ và vận chuyển nhanh. Doanh nghiệp nên cân bằng đội xe theo nhu cầu thực tế: duy trì container cho các đơn hàng lớn, định kỳ và sử dụng xe tải 15T cho các đơn hàng linh hoạt, giao hàng khẩn. Trong mùa cao điểm (Tết), cần tăng cường cả hai loại phương tiện để không bỏ lỡ nhu cầu thị trường.

# ### CELL 9: Biểu đồ 4 - Scatter Plot (Tương quan Actual_Trips vs Total_Volume_Tons)

# In[9]:


# ============================================================
# CELL 9: BIỂU ĐỒ 4 - SCATTER PLOT
# Thể hiện mối tương quan giữa Số chuyến xe (Actual_Trips)
# và Tổng khối lượng hàng hóa (Total_Volume_Tons).
# Mục tiêu: Kiểm tra xem 2 biến này có quan hệ tuyến tính không.
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Subplot 1: Scatter Plot tổng thể ---
# Tô màu theo Is_Peak_Event để phân biệt tuần thường và tuần lễ tết
colors_map = {0: '#3498DB', 1: '#E74C3C'}
labels_map = {0: 'Tuần thường', 1: 'Tuần cao điểm lễ tết'}

for peak_val, group in df.groupby('Is_Peak_Event'):
    axes[0].scatter(
        group['Actual_Trips'],
        group['Total_Volume_Tons'],
        c=colors_map[peak_val],
        label=labels_map[peak_val],
        alpha=0.6,
        s=50,
        edgecolors='white',
        linewidths=0.5
    )

# Thêm đường hồi quy tuyến tính (Regression Line)
from numpy.polynomial.polynomial import polyfit
x_all = df['Actual_Trips'].values
y_all = df['Total_Volume_Tons'].values
coeffs = np.polyfit(x_all, y_all, 1)  # Hồi quy bậc 1
x_line = np.linspace(x_all.min(), x_all.max(), 100)
y_line = np.polyval(coeffs, x_line)
axes[0].plot(x_line, y_line, color='#2ECC71', linewidth=2.5,
             linestyle='-', label=f'Đường hồi quy tuyến tính\ny = {coeffs[0]:.2f}x + {coeffs[1]:.2f}')

# Tính hệ số tương quan Pearson
corr_coef = np.corrcoef(x_all, y_all)[0, 1]

axes[0].set_title(f'Tương Quan: Số Chuyến Xe vs Khối Lượng Hàng\n(Hệ số tương quan Pearson r = {corr_coef:.4f})',
                  fontsize=13, fontweight='bold')
axes[0].set_xlabel('Số Chuyến Xe (Actual_Trips)')
axes[0].set_ylabel('Tổng Khối Lượng Hàng (Tấn)')
axes[0].legend(loc='upper left')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# --- Subplot 2: Scatter Plot phân theo Loại xe (Fleet_Type) ---
colors_fleet_map = {'Truck_15T': '#9B59B6', 'Container_30T': '#E67E22'}

for fleet, group in df.groupby('Fleet_Type'):
    axes[1].scatter(
        group['Actual_Trips'],
        group['Total_Volume_Tons'],
        c=colors_fleet_map[fleet],
        label=fleet,
        alpha=0.6,
        s=55,
        edgecolors='white',
        linewidths=0.5
    )

# Tính và vẽ đường hồi quy cho từng loại xe
for fleet, group in df.groupby('Fleet_Type'):
    coeffs_f = np.polyfit(group['Actual_Trips'].values, group['Total_Volume_Tons'].values, 1)
    x_f = np.linspace(group['Actual_Trips'].min(), group['Actual_Trips'].max(), 100)
    y_f = np.polyval(coeffs_f, x_f)
    axes[1].plot(x_f, y_f, color=colors_fleet_map[fleet], linewidth=2, linestyle='--')

axes[1].set_title('Tương Quan Phân Theo Loại Xe\n(Truck_15T vs Container_30T)',
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('Số Chuyến Xe (Actual_Trips)')
axes[1].set_ylabel('Tổng Khối Lượng Hàng (Tấn)')
axes[1].legend(loc='upper left')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

plt.tight_layout(pad=3.0)
plt.savefig('05_ScatterPlot_TuongQuan.png', bbox_inches='tight', dpi=150)
plt.show()
print(f'📊 Hệ số tương quan Pearson (r) = {corr_coef:.4f}')
print(f'📊 Hệ số xác định (R²) = {corr_coef**2:.4f}')
print('✅ Đã lưu biểu đồ: 05_ScatterPlot_TuongQuan.png')


# ### 📝 NHẬN XÉT & INSIGHT - BIỂU ĐỒ 4 (SCATTER PLOT)
# 
# > **Mối tương quan chặt chẽ:** Biểu đồ phân tán cho thấy tồn tại một **mối tương quan dương mạnh (Positive Correlation)** giữa số chuyến xe (`Actual_Trips`) và tổng khối lượng hàng hóa vận chuyển (`Total_Volume_Tons`). Hệ số tương quan Pearson (r) tiệm cận 1.0 cho thấy khi số chuyến xe tăng lên, tổng khối lượng hàng hóa cũng tăng tỷ lệ thuận — điều này hợp lý về mặt vận hành logistics.
# >
# > **Phân biệt theo loại xe:** Biểu đồ phải (phân theo loại xe) cho thấy hai đám mây điểm của **Truck_15T** và **Container_30T** nằm ở các vùng giá trị khác nhau. Container_30T có xu hướng tạo ra khối lượng vận chuyển cao hơn trên mỗi chuyến, trong khi Truck_15T có lượng chuyến nhiều hơn nhưng khối lượng mỗi chuyến thấp hơn. Đường hồi quy tuyến tính của từng loại xe đều cho thấy độ dốc dương, xác nhận mối tương quan thuận chiều trên cả hai phân khúc.
# >
# > **Ứng dụng thực tiễn:** Kết quả này cho phép xây dựng mô hình dự báo đơn giản: từ kế hoạch vận tải (số chuyến) có thể ước lượng được khối lượng hàng hóa tương ứng, hỗ trợ đội kinh doanh lập báo giá và phân bổ nguồn lực hiệu quả hơn.

# ---
# ## 📉 NHIỆM VỤ 3: PHÂN RÃ CHUỖI THỜI GIAN (TIME SERIES DECOMPOSITION)

# ### CELL 10: Phân rã chuỗi thời gian theo tháng

# In[10]:


# ============================================================
# CELL 10: PHÂN RÃ CHUỖI THỜI GIAN (TIME SERIES DECOMPOSITION)
# Sử dụng: statsmodels.tsa.seasonal.seasonal_decompose
#
# Mô hình phân rã: Dữ liệu thực tế = Trend + Seasonality + Residual
#
# --- Giải thích tham số 'period' ---
# Dữ liệu đã được gộp theo THÁNG. Chúng ta có 5 năm dữ liệu.
# Tính mùa vụ lặp lại theo chu kỳ 1 NĂM = 12 THÁNG.
# => Thiết lập: period = 12
# Điều này có nghĩa là hàm seasonal_decompose sẽ tìm kiếm
# và tách ra một mẫu mùa vụ lặp đi lặp lại mỗi 12 điểm dữ liệu
# (tức mỗi 12 tháng = 1 năm), rất phù hợp để phát hiện
# tính mùa vụ Tết Nguyên Đán và các cao điểm lễ lớn trong năm.
# ============================================================

print('=' * 60)
print('📉 PHÂN RÃ CHUỖI THỜI GIAN - TỔNG KHỐI LƯỢNG HÀNG HÓA')
print('=' * 60)

# Chuẩn bị dữ liệu: chọn cột Total_Volume, đặt Date làm index
ts_data = df_monthly.set_index('Date')['Total_Volume']

# Đảm bảo index có tần suất tháng (Monthly frequency)
ts_data = ts_data.asfreq('MS')  # MS = Month Start

print(f'  • Số tháng dữ liệu: {len(ts_data)}')
print(f'  • Giai đoạn: từ {ts_data.index[0].strftime("%m/%Y")} đến {ts_data.index[-1].strftime("%m/%Y")}')
print(f'  • Chu kỳ mùa vụ (period): 12 tháng (1 năm)')
print()

# Thực hiện phân rã chuỗi thời gian
# model='additive': phù hợp khi biên độ mùa vụ không thay đổi theo mức độ của xu hướng
# model='multiplicative': dùng khi biên độ mùa vụ tỷ lệ thuận với xu hướng
# => Chọn 'additive' vì biên độ biến động mùa vụ trong vận tải khá ổn định
decomposition = seasonal_decompose(
    ts_data,
    model='additive',   # Mô hình cộng (Additive Model)
    period=12,          # Chu kỳ mùa vụ = 12 tháng = 1 năm
    extrapolate_trend='freq'  # Ngoại suy xu hướng tại 2 đầu chuỗi (tránh giá trị NaN)
)

# Lấy 3 thành phần
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# --- Vẽ biểu đồ 4 thành phần (Original + Trend + Seasonal + Residual) ---
fig, axes = plt.subplots(4, 1, figsize=(16, 16))

# Màu cho từng thành phần
plot_colors = ['#2E86AB', '#E74C3C', '#2ECC71', '#95A5A6']
plot_data = [
    (ts_data, 'Dữ Liệu Gốc (Original)', 'Khối lượng (Tấn)'),
    (trend, 'Xu Hướng (Trend)', 'Xu hướng (Tấn)'),
    (seasonal, 'Mùa Vụ (Seasonality)', 'Giá trị mùa vụ'),
    (residual, 'Nhiễu (Residual)', 'Phần dư (Tấn)')
]

for i, (data, title, ylabel) in enumerate(plot_data):
    axes[i].plot(data, color=plot_colors[i], linewidth=2.0, marker='o', markersize=3)
    if i == 3:  # Biểu đồ nhiễu thêm đường y=0
        axes[i].axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        axes[i].fill_between(data.index, data, 0, alpha=0.3, color=plot_colors[i])
    axes[i].set_title(f'[{i+1}] {title}', fontsize=14, fontweight='bold', pad=8)
    axes[i].set_ylabel(ylabel, fontsize=11)
    axes[i].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    if i < 3:
        axes[i].set_xticklabels([])

axes[-1].set_xlabel('Thời gian (Tháng/Năm)', fontsize=12)

plt.suptitle('Phân Rã Chuỗi Thời Gian - Tổng Khối Lượng Hàng Hóa Vận Chuyển\n(Additive Model | Period = 12 tháng)',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout(pad=2.0)
plt.savefig('06_Decomposition_ChuoiThoiGian.png', bbox_inches='tight', dpi=150)
plt.show()
print('✅ Đã lưu biểu đồ: 06_Decomposition_ChuoiThoiGian.png')


# ### CELL 11: Phân tích chi tiết từng thành phần

# In[11]:


# ============================================================
# CELL 11: Phân tích và in số liệu chi tiết từng thành phần
# ============================================================

print('=' * 60)
print('📊 PHÂN TÍCH CHI TIẾT TỪNG THÀNH PHẦN')
print('=' * 60)

# --- Xu hướng (Trend) ---
print('\n1️⃣  XU HƯỚNG (TREND):')
print(f'   Giá trị đầu kỳ (tháng đầu tiên): {trend.iloc[0]:,.2f} Tấn')
print(f'   Giá trị cuối kỳ (tháng cuối cùng): {trend.iloc[-1]:,.2f} Tấn')
growth_rate = (trend.iloc[-1] - trend.iloc[0]) / trend.iloc[0] * 100
print(f'   Tỷ lệ tăng trưởng tổng thể (5 năm): {growth_rate:+.2f}%')
avg_monthly_growth = growth_rate / (len(trend) - 1)
print(f'   Tăng trưởng trung bình mỗi tháng: {avg_monthly_growth:+.4f}%')

# --- Mùa vụ (Seasonality) ---
print('\n2️⃣  MÙA VỤ (SEASONALITY):')
# Lấy chu kỳ mùa vụ đại diện (12 tháng đầu)
seasonal_cycle = seasonal.iloc[:12].values
month_names = ['Tháng 1', 'Tháng 2', 'Tháng 3', 'Tháng 4', 'Tháng 5', 'Tháng 6',
               'Tháng 7', 'Tháng 8', 'Tháng 9', 'Tháng 10', 'Tháng 11', 'Tháng 12']

seasonal_df = pd.DataFrame({
    'Tháng': month_names[:len(seasonal_cycle)],
    'Giá trị mùa vụ (Tấn)': [f'{v:+,.2f}' for v in seasonal_cycle]
})
print(seasonal_df.to_string(index=False))
print(f'\n   Biên độ dao động mùa vụ: {seasonal.max() - seasonal.min():,.2f} Tấn')
print(f'   Tháng có chỉ số mùa vụ CAO NHẤT: {month_names[np.argmax(seasonal_cycle)]}')
print(f'   Tháng có chỉ số mùa vụ THẤP NHẤT: {month_names[np.argmin(seasonal_cycle)]}')

# --- Nhiễu (Residual) ---
print('\n3️⃣  NHIỄU (RESIDUAL):')
print(f'   Trung bình nhiễu: {residual.mean():,.4f} Tấn (gần 0 là tốt)')
print(f'   Độ lệch chuẩn nhiễu: {residual.std():,.2f} Tấn')
print(f'   Giá trị nhiễu lớn nhất: {residual.max():+,.2f} Tấn')
print(f'   Giá trị nhiễu nhỏ nhất: {residual.min():+,.2f} Tấn')

print('\n✅ Phân tích hoàn tất!')


# ### 📝 NHẬN XÉT & INSIGHT - PHÂN RÃ CHUỖI THỜI GIAN
# 
# > **[1] Xu hướng (Trend):** Thành phần xu hướng cho thấy tổng khối lượng hàng hóa vận chuyển trên trục Bắc–Nam có **xu hướng tăng trưởng liên tục và bền vững** trong giai đoạn 2021–2025. Đây là tín hiệu tích cực phản ánh sự phát triển của hạ tầng logistics, tăng trưởng thương mại điện tử và sản xuất công nghiệp tại Việt Nam. Tốc độ tăng trưởng dương này là nền tảng để doanh nghiệp có thể tự tin mở rộng đội xe và mạng lưới vận hành trong các năm tới.
# >
# > **[2] Tính Mùa Vụ (Seasonality):** Thành phần mùa vụ bộc lộ một **mẫu dao động định kỳ rõ nét theo chu kỳ 12 tháng (1 năm)**. Chỉ số mùa vụ đạt **đỉnh cao nhất vào tháng 1** (trước Tết Nguyên Đán) khi nhu cầu tích trữ hàng hóa tăng vọt, sau đó **giảm sâu nhất vào tháng 2–3** (nghỉ Tết kéo dài, nhà máy đóng cửa). Từ tháng 4, nhu cầu phục hồi dần và duy trì ở mức ổn định đến cuối năm với một đỉnh phụ nhỏ vào tháng 9–10 (mùa hàng cuối năm). Biên độ dao động mùa vụ lớn này khẳng định tầm quan trọng của việc lập kế hoạch vận hành theo mùa, tránh tình trạng thiếu hụt xe vào cao điểm và dư thừa vào thấp điểm.
# >
# > **[3] Nhiễu (Residual):** Phần nhiễu (hay phần dư) biểu thị các yếu tố ngẫu nhiên không giải thích được bởi xu hướng và mùa vụ. Trung bình của nhiễu xấp xỉ 0, điều này cho thấy mô hình phân rã **Additive** đã giải thích tốt hầu hết các biến động trong dữ liệu. Các đỉnh nhiễu bất thường (lớn bất thường) nếu có thể là dấu hiệu của các sự kiện không lường trước như đứt gãy chuỗi cung ứng, thay đổi chính sách vận tải đột ngột, hoặc thiên tai — cần được doanh nghiệp theo dõi và phân tích riêng.

# ---
# ## ✅ TỔNG KẾT
# 
# | Nhiệm vụ | Nội dung | Kết quả |
# |---|---|---|
# | **Nhiệm vụ 1** | Xử lý dữ liệu | ✅ Kiểm tra NaN, Outliers (IQR), Gộp Tuần→Tháng |
# | **Nhiệm vụ 2** | Trực quan hóa | ✅ Line Chart, Bar Chart, Pie Chart, Scatter Plot |
# | **Nhiệm vụ 3** | Phân rã chuỗi thời gian | ✅ Trend + Seasonality + Residual (period=12) |
# 
# **Các file biểu đồ đã được lưu:**
# - `01_Outlier_Boxplot.png` — Kiểm tra ngoại lai
# - `02_LineChart_XuHuong.png` — Xu hướng theo thời gian
# - `03_BarChart_SoSanh.png` — So sánh theo tuyến/tháng
# - `04_PieChart_LoaiXe.png` — Tỷ trọng theo loại xe
# - `05_ScatterPlot_TuongQuan.png` — Tương quan chuyến-khối lượng
# - `06_Decomposition_ChuoiThoiGian.png` — Phân rã chuỗi thời gian
# 
# ---
# *Phân tích thực hiện bằng Python | Thư viện: pandas, numpy, matplotlib, seaborn, statsmodels*
