# Dự đoán và phân khúc khách hàng rời bỏ (Churn Prediction) – E-commerce

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/ML_final_project.ipynb)

Dự án Machine Learning phân tích hành vi khách hàng của một công ty thương mại điện tử nhằm:

1. **EDA** – tìm ra đặc điểm của những khách hàng đã rời bỏ và đề xuất giải pháp giảm churn.
2. **Supervised Learning** – xây dựng mô hình dự đoán khách hàng sẽ rời bỏ.
3. **Unsupervised Learning** – phân khúc nhóm khách hàng đã rời bỏ (KMeans) để thiết kế khuyến mãi phù hợp cho từng nhóm.

## Cấu trúc thư mục

```
.
├── raws/
│   └── churn_prediction.xlsx  # Dữ liệu thô (5.630 khách hàng, 20 cột)
├── ML_final_project.ipynb     # Notebook chính: tiền xử lý, EDA, mô hình, phân cụm
├── requirements.txt
└── README.md
```

## Dữ liệu

5.630 khách hàng, tỷ lệ churn **16,8%** (948 khách hàng). Một số cột chính:

| Cột | Ý nghĩa |
|---|---|
| `Churn` | Nhãn rời bỏ (1 = rời bỏ) |
| `Tenure` | Thời gian gắn bó với công ty |
| `PreferredLoginDevice` | Thiết bị đăng nhập ưa thích |
| `CityTier` | Phân hạng thành phố (1, 2, 3) |
| `WarehouseToHome` | Khoảng cách từ kho đến nhà khách hàng |
| `PreferredPaymentMode` | Phương thức thanh toán ưa thích |
| `HourSpendOnApp` | Số giờ sử dụng app/website |
| `PreferedOrderCat` | Danh mục mua nhiều nhất tháng trước |
| `SatisfactionScore` | Điểm hài lòng về dịch vụ |
| `NumberOfAddress` | Số địa chỉ đã thêm |
| `Complain` | Có khiếu nại trong tháng trước hay không |
| `OrderCount`, `CouponUsed`, `DaySinceLastOrder`, `CashbackAmount` | Hành vi mua hàng |

## Quy trình

**Tiền xử lý**
- Điền giá trị thiếu (~4-5% ở 7 cột): median cho `tenure`, `warehousetohome`, `orderamounthikefromlastyear`, `ordercount`, `daysincelastorder`; 0 cho `hourspendonapp`, `couponused`.
- Chuẩn hoá các giá trị cùng nghĩa: `Phone` → `Mobile Phone`, `COD` → `Cash on Delivery`, `CC` → `Credit Card`, `Mobile` → `Mobile Phone`.
- One-hot encoding các biến phân loại; loại `warehousetohome` và các giá trị `numberofaddress` bất thường (19-22) khi huấn luyện mô hình.

**Mô hình** – chia Train/Validation/Test = 70/15/15 (stratified), so sánh Logistic Regression, Decision Tree, Random Forest, Gradient Boosting theo *balanced accuracy* (do dữ liệu mất cân bằng), sau đó tinh chỉnh Random Forest bằng `GridSearchCV` (5-fold).

**Phân cụm** – lọc khách hàng `churn == 1`, Min-Max scaling, chọn K bằng Elbow + Silhouette, áp dụng KMeans và mô tả đặc điểm từng cụm.

## Kết quả

### 1. Hành vi của khách hàng rời bỏ
- **Khách hàng mới** (tenure thấp) có tỷ lệ churn cao nhất.
- Khách hàng churn **nhận ít cashback hơn**.
- Tỷ lệ churn ở nhóm **có khiếu nại** cao hơn rõ rệt.

→ Đề xuất: đẩy mạnh ưu đãi và trải nghiệm onboarding cho khách mới, tăng cashback, xử lý khiếu nại triệt để.

### 2. Mô hình dự đoán churn

| Mô hình (validation) | Precision | Recall | Balanced accuracy |
|---|---|---|---|
| **Random Forest** | 0.95 | 0.82 | **0.91** |
| Gradient Boosting | 0.81 | 0.64 | 0.81 |
| Decision Tree | 0.67 | 0.59 | 0.77 |
| Logistic Regression | 0.73 | 0.56 | 0.76 |

Random Forest sau khi tinh chỉnh (`n_estimators=200, max_depth=30, bootstrap=False`):

| Tập | Balanced accuracy |
|---|---|
| Validation | 0.944 |
| **Test** | **0.925** (recall lớp churn 0.86, precision 0.95) |

> Balanced accuracy trên tập train = 1.0 → mô hình có dấu hiệu overfitting, có thể cải thiện thêm bằng regularization.

### 3. Phân khúc khách hàng rời bỏ (KMeans, K = 4)

| Cụm | Đặc điểm | Đề xuất |
|---|---|---|
| 0 – Khách lâu năm, giá trị cao | Tenure cao, nhiều đơn, cashback cao, ít khiếu nại, hài lòng thấp | Chương trình khách hàng thân thiết, ưu đãi Laptop & Accessory |
| 1 – Khách mới, ít tương tác | Tenure thấp nhất, ít đơn, cashback thấp, mua điện thoại | Ưu đãi onboarding, voucher đơn hàng tiếp theo |
| 2 – Khách mới hay khiếu nại | Khiếu nại cao nhất (~69%), ít đơn | Chủ động xử lý khiếu nại, voucher bồi thường |
| 3 – Khách ở tỉnh xa | City tier 3, xa kho nhất, khiếu nại cao | Miễn/giảm phí vận chuyển, giao hàng nhanh hơn |

> Silhouette score chỉ ~0.14 → các cụm không tách biệt rõ, kết quả phân cụm mang tính tham khảo.

## Cách chạy

### Google Colab (khuyến nghị)

1. Bấm nút **Open In Colab** ở đầu trang.
2. Chọn **Runtime → Run all**.

Không cần upload dữ liệu: notebook tự tải `churn_prediction.xlsx` từ Google Drive vào thư mục `raws/`. Các thư viện cần dùng đều có sẵn trên Colab.

### Chạy local

```bash
git clone https://github.com/hoangvanhieu2004/Churn-Prediction-E-commerce-ML.git
cd YOUR_REPO
pip install -r requirements.txt
jupyter notebook ML_final_project.ipynb
```

Notebook đọc dữ liệu từ `raws/churn_prediction.xlsx`; nếu chưa có file, notebook sẽ tự tải về bằng `gdown`.

## Công nghệ

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · missingno
