"""
Đề tài: Dự đoán nguy cơ nhiễm trùng huyết trong quá trình sàng lọc bệnh nhân tại phòng cấp cứu:
1. Sử dụng các mô hình học máy để dự đoán và so sánh hiệu suất các mô hình.
2. Giải thích dự đoán bằng XAI (LIME/SHAP)

Sinh viên: Nguyễn Bảo Trúc - 20227186
"""

# Chạy lệnh dưới ở Terminal trước khi chạy code:
# pip install pandas numpy openpyxl scikit-learn imbalanced-learn xgboost lightgbm catboost matplotlib lime shap traceback os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import traceback
import os

# Tiền xử lý và Lấy mẫu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

# Các mô hình
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier

# Các chỉ số đánh giá
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, accuracy_score,
    confusion_matrix, classification_report, brier_score_loss
)

# Giải thích mô hình
import lime
import lime.lime_tabular
import shap

# Bỏ qua một số cảnh báo để output gọn gàng hơn
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')
warnings.filterwarnings("ignore", category=FutureWarning) # Tắt cảnh báo liên quan đến phiên bản shap sắp tới

# --- Hằng số ---
os.chdir('C:/Users/LENOVO/Downloads') # THAY ĐỔI ĐƯỜNG DẪN ĐẾN THƯ MỤC LÀM VIỆC CỦA BẠN
FILE_PATH = './data/triage_data.xlsx' # Đường dẫn file dữ liệu
TARGET_COLUMN = 'sepsis' # Tên cột mục tiêu
COLUMNS_TO_DROP = ['subject_id', 'stay_id', 'hadm_id', # Các cột cần loại bỏ
                   'ed_intime', 'ed_outtime', 'icu_stay_id', 'icu_intime',
                   'chiefcomplaint', 'chiefcomplaint_changed']
RANDOM_STATE = 42 # Giá trị seed cho các quá trình ngẫu nhiên để đảm bảo tính lặp lại
TEST_SIZE = 0.2 # Tỷ lệ tập kiểm tra
EXPLAIN_SAMPLE_INDEX = 150 # CHỌN CHỈ SỐ MẪU TRONG TẬP TEST ĐỂ GIẢI THÍCH BẰNG LIME/SHAP
SHAP_BACKGROUND_SAMPLE_SIZE = 100 # Số lượng mẫu nền cho KernelExplainer (nếu sử dụng)

# --- Các hàm ---

def load_and_preprocess_data(file_path, target_col, cols_to_drop, test_size, random_state):
    """
    Tải dữ liệu, chọn đặc trưng, thực hiện SMOTE, chia tập huấn luyện/kiểm tra và chuẩn hóa dữ liệu.
    """
    print("Đang tải và tiền xử lý dữ liệu...")
    try:
        data = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp tại {file_path}")
        return None
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {e}")
        return None

    predictors = data.columns.difference(cols_to_drop + [target_col])
    if target_col not in data.columns:
        print(f"Lỗi: Không tìm thấy cột mục tiêu '{target_col}' trong dữ liệu.")
        return None

    X = data[predictors]
    y = data[target_col].astype(int)
    print("\nPhân phối lớp trước khi SMOTE oversampling:")
    print(y.value_counts(normalize=True))

    print("\nThực hiện SMOTE Oversampling...")
    minority_class_count = y.value_counts().min()
    k_neighbors_smote = min(5, minority_class_count - 1) if minority_class_count > 1 else 1

    if k_neighbors_smote < 1 and minority_class_count > 0 :
        print(f"Cảnh báo: Lớp thiểu số chỉ có {minority_class_count} mẫu. SMOTE có thể không hiệu quả hoặc k_neighbors cần là 1.")
        if minority_class_count <= 1:
            print(f"Lỗi: Lớp thiểu số có {minority_class_count} mẫu. Không thể áp dụng SMOTE. Kiểm tra dữ liệu.")
            return None
        k_neighbors_smote = max(1, minority_class_count -1)

    print(f"  Sử dụng k_neighbors = {k_neighbors_smote} cho SMOTE (đã điều chỉnh nếu lớp thiểu số nhỏ).")
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors_smote if k_neighbors_smote > 0 else 1)
    try:
        X_resampled, y_resampled = smote.fit_resample(X, y)
    except ValueError as e:
        print(f"Lỗi trong quá trình SMOTE oversampling: {e}.")
        print("Điều này có thể do một lớp có quá ít mẫu cho tham số k_neighbors của SMOTE, ngay cả sau khi điều chỉnh.")
        print(f"Số lượng mẫu lớp thiểu số: {minority_class_count}, k_neighbors đã sử dụng: {k_neighbors_smote}")
        return None
    except Exception as e:
        print(f"Lỗi không mong muốn xảy ra trong SMOTE: {e}")
        traceback.print_exc()
        return None

    print("\nPhân phối lớp sau khi SMOTE oversampling:")
    print(pd.Series(y_resampled).value_counts(normalize=True))

    print("\nChia dữ liệu thành tập huấn luyện/kiểm tra...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=test_size, random_state=random_state, stratify=y_resampled
    )

    print("Chuẩn hóa đặc trưng...")
    scaler = StandardScaler()
    X_train_scaled_np = scaler.fit_transform(X_train)
    X_test_scaled_np = scaler.transform(X_test)

    # Chuyển lại thành DataFrame để giữ tên cột (quan trọng cho LIME/SHAP)
    # và giữ index gốc để map lại với dữ liệu gốc nếu cần
    X_train_scaled = pd.DataFrame(X_train_scaled_np, columns=predictors, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_np, columns=predictors, index=X_test.index)
    # Chuyển X_train, X_test (chưa scale) thành DataFrame (nếu chúng là numpy array sau SMOTE)
    X_train = pd.DataFrame(X_train, columns=predictors)
    X_test = pd.DataFrame(X_test, columns=predictors)


    print(f"\nChia dữ liệu: Tập huấn luyện={X_train.shape[0]}, Tập kiểm tra={X_test.shape[0]}")
    print("Hoàn thành tiền xử lý.")

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, predictors, scaler

def define_models(random_state):
    """
    Khởi tạo và trả về một từ điển các mô hình phân loại.
    """
    models = {
        'LogisticRegression': LogisticRegression(random_state=random_state, max_iter=1000, solver='liblinear'),
        'SVM': SVC(probability=True, random_state=random_state, C=1.0),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        'XGBoost': XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=random_state),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=random_state),
        'CatBoost': CatBoostClassifier(n_estimators=100, random_state=random_state, verbose=0),
        'MLPClassifier': MLPClassifier(random_state=random_state, max_iter=500, early_stopping=True, hidden_layer_sizes=(100, 50))
    }
    return models

def find_best_threshold(y_true, y_proba, metric='fbeta', step=0.01, beta=2.0):
    """
    Tìm ngưỡng phân loại tốt nhất dựa trên metric cho trước (ví dụ: F1, balanced accuracy).
    """
    thresholds = np.arange(0, 1 + step, step)
    best_threshold = 0.5
    best_score = -np.inf

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'fbeta': # Mặc định là fbeta, có thể dùng f1
            score = f1_score(y_true, y_pred, beta=beta, zero_division=0) # Sử dụng beta cho F-beta score
        elif metric == 'balanced':
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            sensitivity = tp / (tp + fn) if (tp + fn) else 0
            specificity = tn / (tn + fp) if (tn + fp) else 0
            score = (sensitivity + specificity) / 2
        else:
            raise ValueError(f"Metric không được hỗ trợ để tìm ngưỡng: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = thresh
    return best_threshold, best_score

def find_best_threshold_cv(model_prototype, X_train_scaled, y_train, n_splits=5, threshold_metric='fbeta', random_state=RANDOM_STATE):
    """
    Tìm ngưỡng phân loại tối ưu sử dụng cross-validation trên tập huấn luyện.
    Trả về ngưỡng tối ưu và điểm số của metric đó trên các fold validation.
    """
    print(f"\nTìm ngưỡng tối ưu bằng {n_splits}-fold Cross-Validation (metric: {threshold_metric})...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_y_val_true = []
    all_y_val_scores = []

    if not isinstance(X_train_scaled, pd.DataFrame):
        X_train_scaled_df = pd.DataFrame(X_train_scaled)
    else:
        X_train_scaled_df = X_train_scaled

    if not isinstance(y_train, pd.Series):
        y_train_series = pd.Series(y_train)
    else:
        y_train_series = y_train

    fold_num = 1
    for train_idx, val_idx in skf.split(X_train_scaled_df, y_train_series):
        print(f"  CV Fold {fold_num}/{n_splits}...")
        from sklearn.base import clone
        model_fold = clone(model_prototype) # Clone model để mỗi fold huấn luyện một model mới

        X_train_fold, X_val_fold = X_train_scaled_df.iloc[train_idx], X_train_scaled_df.iloc[val_idx]
        y_train_fold, y_val_fold = y_train_series.iloc[train_idx], y_train_series.iloc[val_idx]

        try:
            model_fold.fit(X_train_fold, y_train_fold)
            y_scores_val_fold = model_fold.predict_proba(X_val_fold)[:, 1]

            all_y_val_true.extend(y_val_fold.tolist())
            all_y_val_scores.extend(y_scores_val_fold.tolist())

        except AttributeError:
            print(f"    Cảnh báo: Mô hình {type(model_prototype).__name__} không hỗ trợ predict_proba trong CV fold. Bỏ qua fold này để tìm ngưỡng.")
        except Exception as e:
            print(f"    Lỗi trong CV fold {fold_num} khi tìm ngưỡng: {e}")
        fold_num += 1

    if not all_y_val_scores:
        print("  Không thể thu thập điểm số từ bất kỳ CV fold nào. Sử dụng ngưỡng mặc định 0.5.")
        return 0.5, 0

    all_y_val_true_np = np.array(all_y_val_true)
    all_y_val_scores_np = np.array(all_y_val_scores)

    if len(all_y_val_true_np) == 0 or len(all_y_val_scores_np) == 0:
        print("  Không có dự đoán nào được thu thập từ CV. Sử dụng ngưỡng mặc định 0.5.")
        return 0.5, 0

    optimal_threshold, best_cv_score = find_best_threshold(all_y_val_true_np, all_y_val_scores_np, metric=threshold_metric)
    print(f"  Ngưỡng tối ưu tìm được từ CV: {optimal_threshold:.4f} (đạt {threshold_metric.upper()}={best_cv_score:.4f} trên tập validation gộp của CV)")
    return optimal_threshold, best_cv_score


def train_and_evaluate_model(name, model_prototype, X_train_scaled, y_train, X_test_scaled, y_test, threshold_metric='f1'):
    """
    Huấn luyện mô hình, tìm ngưỡng tối ưu bằng CV, dự đoán trên tập test và in các chỉ số đánh giá.
    """
    print(f"\n--- Huấn luyện và Đánh giá: {name} ---")

    # 1. Tìm ngưỡng tối ưu bằng Cross-Validation trên tập huấn luyện
    optimal_cv_threshold, best_cv_score = find_best_threshold_cv(
        model_prototype, X_train_scaled, y_train, n_splits=5, threshold_metric=threshold_metric
    )

    # 2. Huấn luyện lại mô hình trên toàn bộ tập huấn luyện
    print(f"Huấn luyện lại mô hình {name} trên toàn bộ tập huấn luyện...")
    from sklearn.base import clone
    final_model = clone(model_prototype) # Clone để đảm bảo mô hình mới
    try:
        final_model.fit(X_train_scaled, y_train)
    except Exception as e:
        print(f"Lỗi khi huấn luyện lại mô hình {name}: {e}")
        return None, {}, None

    # 3. Dự đoán xác suất trên tập Test
    try:
        y_scores_test = final_model.predict_proba(X_test_scaled)[:, 1]
    except AttributeError: # Nếu mô hình không có predict_proba
        try:
            # Thử dùng decision_function hoặc predict
            print(f"Cảnh báo: Mô hình {name} không có predict_proba.")
            if hasattr(final_model, 'decision_function'):
                y_decision_test = final_model.decision_function(X_test_scaled)
                # Cần chuẩn hóa decision_function score nếu dùng cho ROC AUC, hoặc chấp nhận kết quả có thể không tối ưu
                # Ở đây, ta sẽ dùng predict() để lấy nhãn 0/1 và coi như y_scores_test là nhãn đó
                print("  Sử dụng decision_function và predict() cho đánh giá.")
                y_pred_test_no_proba = final_model.predict(X_test_scaled)
                y_scores_test = y_pred_test_no_proba # Dùng cho ROC AUC, có thể không lý tưởng
            else:
                y_pred_test_no_proba = final_model.predict(X_test_scaled)
                y_scores_test = y_pred_test_no_proba # Dùng cho ROC AUC
                print("  Sử dụng predict() cho đánh giá.")
            optimal_cv_threshold = 0.5 # Với predict(), ngưỡng thường là 0.5 (mặc định của predict)
        except AttributeError:
            print(f"Lỗi: Mô hình {name} không hỗ trợ predict_proba, decision_function, hoặc predict.")
            return final_model, {}, None # Trả về model đã huấn luyện nhưng không thể dự đoán
    except Exception as e:
        print(f"Lỗi trong quá trình dự đoán trên tập test: {e}")
        return final_model, {}, None

    # Sử dụng ngưỡng đã tìm được từ CV để tạo nhãn dự đoán cuối cùng
    current_threshold_for_pred = optimal_cv_threshold
    if 'y_pred_test_no_proba' in locals(): # Nếu đã dùng predict()
         y_pred_test = y_pred_test_no_proba
         current_threshold_for_pred = "N/A (đã dùng predict())"
    else:
         y_pred_test = (y_scores_test >= optimal_cv_threshold).astype(int)


    # 4. Tính các chỉ số trên tập Test
    accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, zero_division=0)
    recall = recall_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)
    # ROC AUC nên dùng y_scores_test (xác suất hoặc decision scores)
    # Nếu y_scores_test là 0/1 (từ predict()), ROC AUC vẫn tính được nhưng có thể không phản ánh đúng
    roc_auc = roc_auc_score(y_test, y_scores_test if 'y_pred_test_no_proba' not in locals() else y_pred_test)
    brier = brier_score_loss(y_test, y_scores_test if 'y_pred_test_no_proba' not in locals() else y_pred_test)


    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n--- Kết quả trên TẬP TEST (sử dụng ngưỡng từ CV: {current_threshold_for_pred}) ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (PPV)")
    print(f"Recall:    {recall:.4f} (Độ nhạy)")
    print(f"Specificity: {specificity:.4f} (Độ đặc hiệu)")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print(f"Brier Score: {brier:.4f}")

    print("\nMa trận nhầm lẫn (Tập Test):")
    print(cm)
    print("\nBáo cáo phân loại (Tập Test):")
    print(classification_report(y_test, y_pred_test, zero_division=0))

    metrics = {
        'cv_tuned_threshold': optimal_cv_threshold,
        'cv_metric_score_on_val_folds': best_cv_score,
        'accuracy_test': accuracy, 'precision_test': precision, 'recall_test': recall,
        'specificity_test': specificity, 'f1_test': f1, 'roc_auc_test': roc_auc, 'brier_test': brier,
        'tn_test': tn, 'fp_test': fp, 'fn_test': fn, 'tp_test': tp
    }
    return final_model, metrics, y_scores_test

def explain_with_lime(model, scaler, X_train_df, X_test_df, predictors, sample_index):
    """
    Giải thích dự đoán cho một mẫu cụ thể bằng LIME.
    X_train_df, X_test_df là dữ liệu gốc (chưa chuẩn hóa).
    """
    model_type = type(model).__name__
    print(f"\n--- Giải thích bằng LIME cho Mẫu {sample_index} sử dụng {model_type} ---")
    if sample_index < 0 or sample_index >= len(X_test_df):
        print(f"Lỗi: Chỉ số mẫu {sample_index} nằm ngoài giới hạn của tập test (kích thước {len(X_test_df)}).")
        return

    # Khởi tạo LIME explainer (dùng dữ liệu train gốc, chưa chuẩn hóa)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_df.values,
        feature_names=predictors,
        class_names=['No Sepsis', 'Sepsis'],
        mode='classification',
        discretize_continuous=True,
        random_state=RANDOM_STATE
    )

    instance = X_test_df.iloc[sample_index] # Mẫu cần giải thích (từ tập test gốc)

    # Hàm dự đoán xác suất cho LIME: nhận dữ liệu chưa chuẩn hóa, chuẩn hóa rồi dự đoán
    def predict_proba_lime(data_unscaled):
        try:
            if data_unscaled.ndim == 1:
                data_unscaled = data_unscaled.reshape(1, -1)
            data_scaled = scaler.transform(data_unscaled)
            return model.predict_proba(data_scaled)
        except Exception as e:
            print(f"Lỗi trong hàm dự đoán của LIME: {e}")
            return np.array([[0.5, 0.5]] * len(data_unscaled)) # Trả về xác suất mặc định nếu lỗi

    print("Đang giải thích mẫu...")
    try:
        exp = explainer.explain_instance(
            data_row=instance.values,
            predict_fn=predict_proba_lime,
            num_features=10 # Số lượng đặc trưng hiển thị trong giải thích
        )

        try:
            exp.show_in_notebook(show_all=False)
        except Exception as e:
            print(f"Không thể hiển thị LIME trong notebook: {e}. Đang lưu vào file HTML.")
            os.makedirs('./results', exist_ok=True)
            html_filename = f'./results/lime_explanation_{model_type}_sample_{sample_index}.html'
            exp.save_to_file(html_filename)
            print(f"Giải thích LIME đã lưu vào {html_filename}")

        instance_scaled = scaler.transform(instance.values.reshape(1, -1))
        pred_prob = model.predict_proba(instance_scaled)[0]
        print(f"Xác suất dự đoán Sepsis của mô hình (Lớp 1): {pred_prob[1]:.4f}")

    except Exception as e:
        print(f"Lỗi trong quá trình giải thích bằng LIME: {e}")
        traceback.print_exc()

def explain_with_shap(model, X_train_scaled_df, X_test_scaled_df, X_test_df, predictors, sample_index):
    """
    Giải thích dự đoán cho một mẫu cụ thể bằng SHAP.
    X_train_scaled_df, X_test_scaled_df là dữ liệu đã chuẩn hóa.
    X_test_df là dữ liệu gốc (chưa chuẩn hóa) để hiển thị giá trị đặc trưng.
    """
    print(f"\n--- Giải thích bằng SHAP cho Mẫu {sample_index} ---")

    # Kiểm tra đầu vào
    if not isinstance(X_test_scaled_df, pd.DataFrame): print("Lỗi: X_test_scaled_df phải là pandas DataFrame."); return
    if not isinstance(X_train_scaled_df, pd.DataFrame): print("Lỗi: X_train_scaled_df phải là pandas DataFrame."); return
    if not isinstance(X_test_df, pd.DataFrame): print("Lỗi: X_test_df phải là pandas DataFrame."); return
    if sample_index < 0 or sample_index >= len(X_test_scaled_df): print(f"Lỗi: Chỉ số mẫu {sample_index} nằm ngoài giới hạn."); return
    if not hasattr(model, 'predict_proba'): print(f"Lỗi: Mô hình cần có phương thức 'predict_proba'."); return
    if not all(p in X_train_scaled_df.columns for p in predictors): print("Lỗi: Không phải tất cả predictors đều có trong cột của X_train_scaled_df."); return
    if not all(p in X_test_scaled_df.columns for p in predictors): print("Lỗi: Không phải tất cả predictors đều có trong cột của X_test_scaled_df."); return
    if not all(p in X_test_df.columns for p in predictors): print("Lỗi: Không phải tất cả predictors đều có trong cột của X_test_df."); return

    # Chuẩn bị dữ liệu (chỉ lấy các cột predictors)
    X_train_scaled_filtered = X_train_scaled_df[predictors]
    X_test_scaled_filtered = X_test_scaled_df[predictors]
    X_test_original_filtered = X_test_df[predictors]

    target_instance_scaled = X_test_scaled_filtered.iloc[[sample_index]] # Mẫu cần giải thích (đã chuẩn hóa)
    target_instance_original_vals = X_test_original_filtered.iloc[sample_index].values # Giá trị gốc của mẫu

    model_type = type(model).__name__
    is_tree_model = model_type in ['RandomForestClassifier', 'GradientBoostingClassifier',
                                   'XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier',
                                   'DecisionTreeClassifier', 'ExtraTreesClassifier']

    try:
        explainer = None
        print(f"Khởi tạo SHAP Explainer ({'Tree' if is_tree_model else 'Kernel'})...")

        # Khởi tạo Explainer
        if is_tree_model:
            try: # Ưu tiên dùng masker nếu có thể
                masker = shap.maskers.Independent(X_train_scaled_filtered, max_samples=SHAP_BACKGROUND_SAMPLE_SIZE)
                explainer = shap.TreeExplainer(model, masker, feature_perturbation="interventional")
                print("Đã khởi tạo TreeExplainer với Independent masker.")
            except Exception: # Fallback nếu masker lỗi
                 print(f"Fallback: Khởi tạo TreeExplainer trực tiếp với dữ liệu (masker thất bại).")
                 background_data = shap.sample(X_train_scaled_filtered, SHAP_BACKGROUND_SAMPLE_SIZE, random_state=RANDOM_STATE)
                 explainer = shap.TreeExplainer(model, background_data)
        else: # Dùng KernelExplainer cho các mô hình khác
            background_data = shap.sample(X_train_scaled_filtered, SHAP_BACKGROUND_SAMPLE_SIZE, random_state=RANDOM_STATE)
            explainer = shap.KernelExplainer(model.predict_proba, background_data)
            print("Đã khởi tạo KernelExplainer.")

        if explainer is None:
            print("Lỗi: Không thể khởi tạo SHAP explainer.")
            return

        # Tính giá trị SHAP
        print(f"Đang tính giá trị SHAP cho mẫu {sample_index}...")
        shap_values_output = explainer(target_instance_scaled) # Thường trả về Explanation object

        # Trích xuất giải thích SHAP cho Lớp 1 (Sepsis)
        shap_exp_for_class1 = None
        if isinstance(shap_values_output, shap.Explanation):
            if hasattr(shap_values_output, 'values') and isinstance(shap_values_output.values, np.ndarray):
                if shap_values_output.values.ndim == 3: # (samples, features, classes)
                    if shap_values_output.values.shape[0] == 1 and shap_values_output.values.shape[2] >= 2:
                        shap_exp_for_class1 = shap_values_output[0, :, 1] # Lớp 1
                elif shap_values_output.values.ndim == 2: # (samples, features) - Giả định cho lớp 1
                    shap_exp_for_class1 = shap_values_output[0]
            if shap_exp_for_class1 is None: # Nếu không trích xuất được
                 print(f"Lỗi: Cấu trúc Explanation.values không như mong đợi: shape {getattr(shap_values_output, 'values', 'N/A').shape}")
                 return
        else: # Xử lý trường hợp output không phải là Explanation object
             print(f"Lỗi: Output của SHAP không phải là Explanation object (loại: {type(shap_values_output)}).")
             return

        if shap_exp_for_class1 is None:
             print("Lỗi: Không thể lấy giải thích SHAP cho lớp 1.")
             return

        # Thay thế dữ liệu hiển thị bằng giá trị gốc của đặc trưng
        try:
            shap_exp_for_class1.data = target_instance_original_vals
        except Exception as data_err:
            print(f"Cảnh báo: Không thể đặt dữ liệu gốc vào Explanation object: {data_err}. Biểu đồ có thể hiển thị giá trị đã chuẩn hóa.")

        print("Hoàn thành trích xuất giá trị SHAP cho lớp 1.")

        # Vẽ biểu đồ
        sepsis_probability = model.predict_proba(target_instance_scaled)[0, 1]
        print(f"Mẫu {sample_index} - Xác suất dự đoán Sepsis (Lớp 1): {sepsis_probability:.4f}")

        plt.figure(figsize=(10, 8))
        print("Đang tạo biểu đồ thác nước SHAP cho Lớp 1...")
        try:
            shap.plots.waterfall(shap_exp_for_class1, max_display=15, show=False)
            plt.title(f"Giải thích SHAP cho Sepsis (Lớp 1) - {model_type} - Mẫu {sample_index}\nXác suất dự đoán: {sepsis_probability:.4f}")
            plt.tight_layout()

            os.makedirs('./results', exist_ok=True)
            pdf_filename = f'./results/shap_waterfall_{model_type}_sample_{sample_index}_class_1.pdf'
            plt.savefig(pdf_filename, bbox_inches='tight')
            print(f"Biểu đồ thác nước SHAP đã lưu vào {pdf_filename}")
            plt.show()
        except Exception as plot_err:
             print(f"Lỗi khi tạo biểu đồ thác nước SHAP: {plot_err}")
             traceback.print_exc()
             plt.close()

    except ImportError:
        print("Không tìm thấy thư viện SHAP. Vui lòng cài đặt: pip install shap")
    except Exception as e:
        print(f"Lỗi không mong muốn xảy ra trong quá trình giải thích bằng SHAP cho mô hình {model_type}: {e}")
        traceback.print_exc()

def display_global_feature_importance(model, model_name, predictors, X_train_scaled_df=None, top_n=15):
    """
    Hiển thị top N đặc trưng quan trọng nhất toàn cục cho mô hình.
    Sử dụng feature_importances_, coef_ hoặc giá trị SHAP trung bình.
    """
    print(f"\n--- Độ quan trọng Đặc trưng Toàn cục cho {model_name} (Top {top_n}) ---")
    importances = None
    feature_names = np.array(predictors)
    source_text = ""

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        source_text = "model.feature_importances_"
        print(f"Nguồn: {source_text}")
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
        source_text = "model.coef_ (giá trị tuyệt đối)"
        print(f"Nguồn: {source_text}")
    else: # Sử dụng SHAP nếu không có thuộc tính trên
        source_text = "Giá trị SHAP tuyệt đối trung bình"
        print(f"Nguồn: {source_text} (có thể chậm với các mô hình không phải cây)")
        if X_train_scaled_df is None:
            print("Cảnh báo: X_train_scaled_df không được cung cấp cho độ quan trọng toàn cục dựa trên SHAP. Bỏ qua.")
            return

        try:
            model_type = type(model).__name__
            is_tree_model_for_shap = model_type in ['RandomForestClassifier', 'GradientBoostingClassifier',
                                       'XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier',
                                       'DecisionTreeClassifier', 'ExtraTreesClassifier']
            explainer = None
            if is_tree_model_for_shap:
                explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            else:
                background_data = shap.sample(X_train_scaled_df, min(SHAP_BACKGROUND_SAMPLE_SIZE, X_train_scaled_df.shape[0]), random_state=RANDOM_STATE)
                if not hasattr(model, 'predict_proba'):
                    print(f"Cảnh báo: Mô hình {model_name} không có predict_proba, SHAP KernelExplainer có thể không hoạt động như mong đợi.")
                    return
                explainer = shap.KernelExplainer(model.predict_proba, background_data)

            if explainer:
                # Tính SHAP trên tập huấn luyện (có thể dùng một phần để nhanh hơn)
                data_for_shap = X_train_scaled_df
                shap_values_output = explainer(data_for_shap)
                shap_values_for_class1 = None

                if isinstance(shap_values_output, shap.Explanation):
                    if hasattr(shap_values_output, 'values') and isinstance(shap_values_output.values, np.ndarray):
                        if shap_values_output.values.ndim == 3: # (samples, features, classes)
                             shap_values_for_class1 = shap_values_output.values[:, :, 1]
                        elif shap_values_output.values.ndim == 2: # (samples, features)
                             shap_values_for_class1 = shap_values_output.values
                elif isinstance(shap_values_output, list) and len(shap_values_output) == 2: # Output cũ của KernelExplainer.shap_values()
                    shap_values_for_class1 = shap_values_output[1]
                elif isinstance(shap_values_output, np.ndarray) and shap_values_output.ndim == 2: # Output của TreeExplainer.shap_values() cho 1 lớp
                    shap_values_for_class1 = shap_values_output

                if shap_values_for_class1 is not None:
                    importances = np.abs(shap_values_for_class1).mean(axis=0)
                else:
                    print("Cảnh báo: Cấu trúc output SHAP không như mong đợi cho độ quan trọng toàn cục. Bỏ qua.")
                    return
            else:
                print(f"Không thể khởi tạo SHAP explainer cho {model_name}. Bỏ qua độ quan trọng dựa trên SHAP.")
                return
        except ImportError:
            print("Không tìm thấy thư viện SHAP. Không thể tính độ quan trọng toàn cục dựa trên SHAP.")
            return
        except Exception as e:
            print(f"Lỗi khi tính độ quan trọng toàn cục dựa trên SHAP cho {model_name}: {e}")
            traceback.print_exc()
            return

    if importances is not None and len(importances) == len(feature_names):
        sorted_indices = np.argsort(importances)[::-1]
        print(f"{'Đặc trưng':<30} | Độ quan trọng")
        print("-" * 50)
        for i in range(min(top_n, len(feature_names))):
            idx = sorted_indices[i]
            print(f"{feature_names[idx]:<30} | {importances[idx]:.4f}")

        # Vẽ biểu đồ
        plt.figure(figsize=(10, max(6, top_n * 0.4)))
        top_indices = sorted_indices[:top_n]
        top_importances = importances[top_indices]
        top_feature_names = feature_names[top_indices]

        model_color_map = {'RandomForest': 'red', 'XGBoost': 'hotpink', 'LightGBM': 'blue', 'CatBoost': 'brown',
                           'LogisticRegression': 'purple', 'SVM': 'green', 'MLPClassifier': 'cyan'} # Sửa MLP
        color = model_color_map.get(model_name, 'gray')

        plt.barh(range(len(top_importances)), top_importances, align='center', color=color)
        plt.yticks(range(len(top_importances)), top_feature_names)
        plt.xlabel(f"Độ quan trọng ({'Giá trị SHAP TB' if 'SHAP' in source_text else 'Giá trị'})")
        plt.ylabel("Đặc trưng")
        plt.title(f"Top {top_n} Đặc trưng Quan trọng Toàn cục - {model_name}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        os.makedirs('./results', exist_ok=True)
        importance_plot_filename = f'./results/global_importance_{model_name}.pdf'
        try:
            plt.savefig(importance_plot_filename, bbox_inches='tight')
            print(f"Biểu đồ độ quan trọng đặc trưng toàn cục đã lưu vào {importance_plot_filename}")
        except Exception as e:
            print(f"Lỗi khi lưu biểu đồ độ quan trọng toàn cục: {e}")
        plt.show()

    elif importances is not None:
        print(f"Cảnh báo: Số lượng giá trị độ quan trọng ({len(importances)}) và đặc trưng ({len(feature_names)}) không khớp cho {model_name}.")
    else:
        print(f"Không thể xác định độ quan trọng của đặc trưng cho mô hình: {model_name}")

# --- Các hàm cho Phân tích Đường cong Quyết định (DCA) ---
def net_benefit(tp, fp, n, threshold):
    """Tính Net Benefit tại một ngưỡng xác suất."""
    if threshold >= 1.0 or threshold <= 0.0: return -np.inf # Tránh chia cho 0
    benefit = tp / n
    harm = fp / n * (threshold / (1 - threshold))
    return benefit - harm

def decision_curve_analysis(y_true, y_prob, thresholds):
    """Thực hiện tính toán cho Phân tích Đường cong Quyết định."""
    n = len(y_true)
    net_benefits = []
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    for threshold in thresholds:
        if threshold <= 0 or threshold >= 1:
            net_benefits.append(-np.inf)
            continue
        y_pred_threshold = (y_prob >= threshold).astype(int)
        tp = np.sum((y_pred_threshold == 1) & (y_true == 1))
        fp = np.sum((y_pred_threshold == 1) & (y_true == 0))
        nb = net_benefit(tp, fp, n, threshold)
        net_benefits.append(nb)
    return net_benefits

def run_decision_curve_analysis(models_dict, X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Chạy DCA cho tất cả các mô hình và vẽ biểu đồ.
    Hàm này sẽ tự huấn luyện lại các mô hình.
    """
    print("\n--- Đang chạy Phân tích Đường cong Quyết định (DCA) ---")
    thresholds = np.linspace(0.01, 0.99, 99) # Ngưỡng từ 0.01 đến 0.99
    dca_results = {}
    trained_models_for_dca = {}

    for name, model_instance in models_dict.items():
        print(f"Huấn luyện và lấy xác suất cho DCA: {name}")
        from sklearn.base import clone
        model = clone(model_instance) # Clone để đảm bảo huấn luyện model mới
        try:
            model.fit(X_train_scaled, y_train)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            nb = decision_curve_analysis(y_test, y_pred_proba, thresholds)
            if not np.all(np.isinf(nb)) and not np.all(np.isnan(nb)): # Chỉ thêm nếu kết quả hợp lệ
                 dca_results[name] = nb
                 trained_models_for_dca[name] = model
            else:
                 print(f"Cảnh báo: Tính toán Net benefit thất bại cho {name}. Bỏ qua trong biểu đồ DCA.")
        except AttributeError:
            print(f"Cảnh báo: Không thể lấy xác suất cho {name}. Bỏ qua trong DCA.")
        except Exception as e:
             print(f"Lỗi khi xử lý {name} cho DCA: {e}")

    if not dca_results:
        print("Lỗi: Không có mô hình nào có thể được xử lý cho DCA.")
        return None

    # Chiến lược "Không điều trị" và "Điều trị tất cả"
    treat_none = np.zeros_like(thresholds)
    prevalence = np.mean(y_test)
    treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    max_net_benefit = max([max(nb) for nb in dca_results.values() if len(nb) > 0 and not np.all(np.isinf(nb))] + [max(treat_all) if len(treat_all) > 0 else -np.inf])


    # Vẽ biểu đồ DCA
    plt.figure(figsize=(12, 8))
    colors = ['purple', 'green', 'red', 'orange', 'hotpink', 'blue', 'brown', 'cyan', 'magenta', 'yellow', 'black']
    # Giới hạn số màu nếu models_dict lớn hơn
    num_colors_to_use = min(len(colors), len(dca_results))


    for i, (name, nb) in enumerate(dca_results.items()):
        plt.plot(thresholds, nb, label=name, color=colors[i % num_colors_to_use], linewidth=2)

    plt.plot(thresholds, treat_none, 'k--', label='Không điều trị', linewidth=2)
    plt.plot(thresholds, treat_all, 'k:', label='Điều trị tất cả', linewidth=2)

    plt.ylim(-0.035, max_net_benefit + 0.05 if max_net_benefit > -np.inf else 0.1) # Xử lý trường hợp max_net_benefit là -inf
    plt.xlim(0, 1)
    plt.xlabel('Ngưỡng Xác suất', fontsize=12)
    plt.ylabel('Lợi ích Ròng (Net Benefit)', fontsize=12)
    plt.title('Phân tích Đường cong Quyết định (DCA)', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    os.makedirs('./results', exist_ok=True)
    plt.savefig("./results/dca_curve_comparison.pdf", bbox_inches='tight')
    print("\nBiểu đồ DCA đã lưu vào ./results/dca_curve_comparison.pdf")
    plt.show()

    return trained_models_for_dca

# --- Luồng thực thi chính ---
if __name__ == "__main__":
    # 1. Tải và Tiền xử lý Dữ liệu
    preprocess_result = load_and_preprocess_data(
        FILE_PATH, TARGET_COLUMN, COLUMNS_TO_DROP, TEST_SIZE, RANDOM_STATE
    )

    if preprocess_result:
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, predictors, scaler = preprocess_result
    else:
        print("Thoát do lỗi tải/tiền xử lý dữ liệu.")
        exit(1)

    # 2. Định nghĩa các Mô hình
    models_dict = define_models(RANDOM_STATE)

    # 3. Chọn MỘT mô hình để phân tích chi tiết (LIME/SHAP) qua input người dùng
    print("\n--- Chọn Mô hình để Phân tích Chi tiết (LIME/SHAP) ---")
    model_names = list(models_dict.keys())
    print("Các mô hình có sẵn:")
    for i, name in enumerate(model_names):
        print(f"{i + 1}: {name}")

    chosen_model_name = None
    chosen_model_instance = None
    while chosen_model_name is None:
        try:
            choice = input(f"Nhập số (1-{len(model_names)}) của mô hình bạn muốn phân tích chi tiết: ")
            chosen_index = int(choice) - 1
            if 0 <= chosen_index < len(model_names):
                chosen_model_name = model_names[chosen_index]
                chosen_model_instance = models_dict[chosen_model_name]
                print(f"Bạn đã chọn: {chosen_model_name}")
            else:
                print(f"Lựa chọn không hợp lệ. Vui lòng nhập một số từ 1 đến {len(model_names)}.")
        except ValueError:
            print("Đầu vào không hợp lệ. Vui lòng nhập một số.")
        except KeyboardInterrupt:
            print("\nThực thi bị ngắt bởi người dùng.")
            exit()
        except Exception as e:
            print(f"Lỗi không mong muốn xảy ra trong quá trình chọn: {e}")

    if chosen_model_instance is None:
         print("Lỗi: Không thể chọn instance của mô hình. Đang thoát.")
         exit(1)

    # 4. Huấn luyện và Đánh giá Mô hình đã chọn
    trained_model, metrics, y_scores = train_and_evaluate_model(
        chosen_model_name,
        chosen_model_instance,
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        threshold_metric='f1' # Có thể thay đổi metric để tìm ngưỡng, ví dụ 'balanced'
    )

    # Chỉ tiếp tục giải thích nếu huấn luyện và đánh giá thành công
    if trained_model and metrics:
        # 5. Giải thích Mô hình đã chọn bằng LIME
        # LIME dùng dữ liệu gốc (chưa scale) cho training_data và instance, scaler được dùng bên trong predict_fn
        explain_with_lime(
            trained_model,
            scaler,
            X_train, # Dữ liệu train gốc
            X_test,  # Dữ liệu test gốc
            predictors,
            EXPLAIN_SAMPLE_INDEX
        )

        # 6. Giải thích Mô hình đã chọn bằng SHAP
        # SHAP thường dùng dữ liệu đã scale cho explainer và instance
        # X_test (gốc) dùng để hiển thị giá trị đặc trưng thực tế
        explain_with_shap(
            trained_model,
            X_train_scaled,
            X_test_scaled,
            X_test, # Dữ liệu test gốc để hiển thị
            predictors,
            EXPLAIN_SAMPLE_INDEX
        )

        print("\n--- Tính toán Độ quan trọng Đặc trưng Toàn cục cho Mô hình đã chọn ---")
        display_global_feature_importance(
            trained_model,
            chosen_model_name,
            predictors,
            X_train_scaled_df=X_train_scaled, # SHAP cần DataFrame cho dữ liệu huấn luyện
            top_n=15
        )
    else:
        print(f"\nBỏ qua giải thích LIME và SHAP do có vấn đề trong quá trình huấn luyện/đánh giá {chosen_model_name}.")


    # # 7. Chạy Phân tích Đường cong Quyết định (DCA) cho TẤT CẢ các mô hình
    # # Hàm này sẽ tự huấn luyện lại các mô hình bên trong
    # trained_dca_models = run_decision_curve_analysis(
    #     models_dict, # Từ điển gốc với tất cả các mô hình
    #     X_train_scaled,
    #     y_train,
    #     X_test_scaled,
    #     y_test
    # )

    print("\n--- Hoàn thành Thực thi Script ---")
