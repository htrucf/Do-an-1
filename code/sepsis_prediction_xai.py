"""
Đề tài: Dự đoán rủi ro nhiễm trùng huyết (sepsis) với dữ liệu phân loại bệnh nhân tại phòng cấp cứu:
1. Sử dụng các mô hình học máy để dự đoán và so sánh hiệu suất các mô hình.
2. Giải thích dự đoán bằng XAI (LIME/SHAP)
3. Vẽ Decision Curve Analysis (DCA).

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

# Preprocessing and Sampling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Evaluation Metrics
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, accuracy_score,
    confusion_matrix, classification_report, brier_score_loss
)

# Interpretability
import lime
import lime.lime_tabular
import shap

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')
warnings.filterwarnings("ignore", category=FutureWarning) # Tắt cảnh báo liên quan đến phiên bản shap sắp tới

# --- Constants ---
os.chdir('C:/Users/LENOVO/Downloads') # THAY ĐỔI ĐƯỜNG DẪN ĐẾN THƯ MỤC CỦA BẠN
FILE_PATH = './triage_data.xlsx'
TARGET_COLUMN = 'sepsis'
COLUMNS_TO_DROP = ['subject_id', 'CHD', 'CHF', 'diabetes', 'stay_id', 'hadm_id',
                   'ed_intime', 'ed_outtime', 'icu_stay_id', 'icu_intime',
                   'chiefcomplaint', 'chiefcomplaint_changed']
RANDOM_STATE = 42
TEST_SIZE = 0.2
EXPLAIN_SAMPLE_INDEX = 150 # CHỌN INDEX MẪU DỮ LIỆU TRONG TẬP TEST ĐỂ GIẢI THÍCH BẰNG LIME/SHAP
SHAP_BACKGROUND_SAMPLE_SIZE = 100 # Số lượng mẫu dùng cho background của KernelExplainer (nếu dùng)

# --- Functions ---

def load_and_preprocess_data(file_path, target_col, cols_to_drop, test_size, random_state):
    """
    Tải dữ liệu, chọn predictors, thực hiện undersampling, tách và scale dữ liệu.
    """
    print("Loading and preprocessing data...")
    try:
        data = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    # Xác định predictors
    predictors = data.columns.difference(cols_to_drop + [target_col])
    if target_col not in data.columns:
        print(f"Error: Target column '{target_col}' not found in the data.")
        return None
    print(f"\nPredictors selected ({len(predictors)}): {list(predictors)}")

    X = data[predictors]
    y = data[target_col].astype(int)

    # Kiểm tra số lượng lớp trước khi sampling
    print("\nClass distribution before undersampling:")
    print(y.value_counts(normalize=True))

    # Thực hiện RandomUnderSampler
    print("\nPerforming Random Undersampling...")
    rus = RandomUnderSampler(random_state=random_state)
    try:
        X_resampled, y_resampled = rus.fit_resample(X, y)
    except ValueError as e:
        print(f"Error during undersampling: {e}. Check class distribution.")
        return None

    print("\nClass distribution after undersampling:")
    print(y_resampled.value_counts(normalize=True))

    # Tách dataset
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=test_size, random_state=random_state, stratify=y_resampled
    )

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    # Fit scaler trên train data và transform cả train và test
    X_train_scaled_np = scaler.fit_transform(X_train)
    X_test_scaled_np = scaler.transform(X_test)

    # Chuyển lại thành DataFrame để giữ tên cột (quan trọng cho LIME/SHAP)
    X_train = pd.DataFrame(X_train, columns=predictors)
    X_test = pd.DataFrame(X_test, columns=predictors)
    # Quan trọng: Giữ index gốc khi tạo DataFrame mới từ numpy array đã scale
    X_train_scaled = pd.DataFrame(X_train_scaled_np, columns=predictors, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_np, columns=predictors, index=X_test.index)

    print(f"\nData split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    print("Preprocessing complete.")

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, predictors, scaler

def define_models(random_state):
    """
    Khởi tạo và trả về một dictionary các mô hình classification.
    """
    models = {
        'LogisticRegression': LogisticRegression(random_state=random_state, max_iter=1000, solver='liblinear'),
        'SVM': SVC(probability=True, random_state=random_state, C=1.0), # Thêm C để tránh warning
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        'XGBoost': XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=random_state),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=random_state),
        'CatBoost': CatBoostClassifier(n_estimators=100, random_state=random_state, verbose=0)
    }
    return models

def train_and_evaluate_model(name, model, X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Huấn luyện mô hình, dự đoán trên tập test và in các metrics đánh giá.
    """
    print(f"\n--- Training and Evaluating: {name} ---")
    try:
        model.fit(X_train_scaled, y_train)
    except Exception as e:
        print(f"Error training model {name}: {e}")
        return None, {}, None # Trả về giá trị không hợp lệ nếu lỗi

    try:
        y_pred = model.predict(X_test_scaled)
        y_scores = model.predict_proba(X_test_scaled)[:, 1]
    except AttributeError:
        print("Warning: predict_proba not available, trying decision_function.")
        try:
            y_decision = model.decision_function(X_test_scaled)
            # Đơn giản hóa: dùng predict làm score nếu decision_function không chuẩn hóa
            y_scores = y_pred
            roc_auc = float('nan')
            brier = float('nan')
            print("Cannot calculate ROC AUC or Brier Score accurately without probabilities.")
        except AttributeError:
            print("Error: Neither predict_proba nor decision_function available.")
            return model, {}, None # Model đã train nhưng không có score
    except Exception as e:
        print(f"Error during prediction or probability calculation: {e}")
        return model, {}, None

    # Tính Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    if 'roc_auc' not in locals(): # Tính nếu chưa bị gán nan ở trên
       roc_auc = roc_auc_score(y_test, y_scores)
       brier = brier_score_loss(y_test, y_scores)

    cm = confusion_matrix(y_test, y_pred)
    # Đảm bảo cm có đủ 4 phần tử trước khi ravel
    if cm.size == 1: # Trường hợp chỉ có 1 lớp được dự đoán (hiếm)
        if y_test.nunique() == 1: # Chỉ có 1 lớp trong y_test
             if y_test.iloc[0] == 0: tn, fp, fn, tp = cm.item(), 0, 0, 0
             else: tn, fp, fn, tp = 0, 0, 0, cm.item()
        else: # Lỗi phân loại hoàn toàn
             print("Warning: Confusion matrix calculation issue (possibly all predictions are the same class). Metrics might be misleading.")
             # Gán giá trị mặc định hoặc xử lý khác
             tn, fp, fn, tp = 0,0,0,0 # Cần xem xét lại logic này nếu xảy ra thường xuyên
    elif cm.size == 4:
         tn, fp, fn, tp = cm.ravel()
    else:
         print(f"Warning: Unexpected confusion matrix shape: {cm.shape}. Setting specificity to NaN.")
         tn, fp, fn, tp = 0,0,0,0 # Hoặc tính toán phù hợp nếu biết cấu trúc cm
         specificity = float('nan')

    if 'specificity' not in locals(): # Tính nếu chưa bị gán nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # In kết quả
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (PPV)")
    print(f"Recall:    {recall:.4f} (Sensitivity)")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print(f"Brier Score: {brier:.4f}")

    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    metrics = {
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'specificity': specificity, 'f1': f1, 'roc_auc': roc_auc, 'brier': brier,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }
    return model, metrics, y_scores # Trả về model đã huấn luyện và scores

def explain_with_lime(model, scaler, X_train_df, X_test_df, predictors, sample_index):
    """
    Giải thích dự đoán cho một mẫu cụ thể bằng LIME.
    """
    model_type = type(model).__name__
    print(f"\n--- LIME Explanation for Sample Index: {sample_index} using {model_type} ---")
    if sample_index < 0 or sample_index >= len(X_test_df):
        print(f"Error: Sample index {sample_index} is out of bounds for the test set (size {len(X_test_df)}).")
        return

    # Khởi tạo LIME explainer (dùng dữ liệu train gốc)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_df.values, # Dữ liệu gốc (chưa scale)
        feature_names=predictors,
        class_names=['Not Sepsis', 'Sepsis'],
        mode='classification',
        discretize_continuous=True,
        random_state=RANDOM_STATE # Để kết quả LIME ổn định hơn
    )

    # Lấy mẫu cần giải thích (từ tập test gốc)
    instance = X_test_df.iloc[sample_index]

    # Tạo hàm dự đoán xác suất tương thích với LIME
    # Nhận numpy array (chưa scale), scale lại, rồi predict_proba
    def predict_proba_lime(data_unscaled):
        try:
            # Đảm bảo input là 2D array cho scaler
            if data_unscaled.ndim == 1:
                data_unscaled = data_unscaled.reshape(1, -1)
            data_scaled = scaler.transform(data_unscaled)
            return model.predict_proba(data_scaled)
        except Exception as e:
            print(f"Error in LIME predict function: {e}")
            # Trả về giá trị mặc định hoặc raise lỗi tùy theo tình huống
            # Ví dụ: trả về phân phối đều nếu không predict được
            return np.array([[0.5, 0.5]] * len(data_unscaled))

    print("Explaining instance...")
    try:
        exp = explainer.explain_instance(
            data_row=instance.values,
            predict_fn=predict_proba_lime,
            num_features=10
        )

        # Hiển thị/lưu giải thích
        try:
            exp.show_in_notebook(show_all=False)
        except Exception as e:
            print(f"Could not show LIME in notebook: {e}. Saving to HTML.")
            html_filename = f'./lime_explanation_{model_type}_sample_{sample_index}.html'
            exp.save_to_file(html_filename)
            print(f"LIME explanation saved to {html_filename}")

        # In thông tin bệnh nhân (subject_id, hadm_id và stay_id)

        # Lấy xác suất dự đoán (phải scale mẫu trước)
        instance_scaled = scaler.transform(instance.values.reshape(1, -1))
        pred_prob = model.predict_proba(instance_scaled)[0]
        print(f"Model Prediction Probability for Sepsis (Class 1): {pred_prob[1]:.4f}")

    except Exception as e:
        print(f"Error during LIME explanation: {e}")
        import traceback
        traceback.print_exc()

# def explain_with_shap(model, X_train_scaled_df, X_test_scaled_df, X_test_df, predictors, sample_index):
#     """
#     Giải thích dự đoán cho một mẫu cụ thể bằng SHAP.
#     Tự động chọn explainer, lấy SHAP values cho class 1 và vẽ waterfall plot.
#     """
#     print(f"\n--- SHAP Explanation for Sample Index: {sample_index} ---")

#     # --- Input Validation ---
#     if not isinstance(X_test_scaled_df, pd.DataFrame):
#         print("Error: X_test_scaled_df must be a pandas DataFrame.")
#         return
#     if sample_index < 0 or sample_index >= len(X_test_scaled_df):
#         print(f"Error: Sample index {sample_index} is out of bounds for the test set (size {len(X_test_scaled_df)}).")
#         return
#     if not hasattr(model, 'predict_proba'):
#          print(f"Error: Model {type(model).__name__} does not have a 'predict_proba' method, which is required for SHAP explanations focused on probability.")
#          return

#     model_type = type(model).__name__
#     is_tree_model = model_type in ['RandomForestClassifier', 'GradientBoostingClassifier',
#                                    'XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier']
#     target_instance = X_test_scaled_df.iloc[[sample_index]] # Lấy mẫu cần giải thích
#     target_instance_original = X_test_df.iloc[[sample_index]]

#     try:
#         explainer = None
#         shap_values_output = None # Biến lưu kết quả từ explainer

#         # --- Initialize Explainer ---
#         if is_tree_model:
#             try:
#                 # Ưu tiên dùng masker nếu data là DataFrame (xử lý tên cột tốt hơn)
#                 masker = shap.maskers.Independent(X_train_scaled_df, max_samples=SHAP_BACKGROUND_SAMPLE_SIZE)
#                 explainer = shap.TreeExplainer(model, masker, feature_names=predictors)
#             except Exception: # Fallback nếu cách trên lỗi (có thể do phiên bản SHAP cũ)
#                  print("Fallback: Initializing TreeExplainer directly with data.")
#                  explainer = shap.TreeExplainer(model, X_train_scaled_df, feature_names=predictors)
#         else:
#             # Dùng background data nhỏ
#             # Lưu ý: sample có thể làm thay đổi nhẹ kết quả SHAP mỗi lần chạy nếu không set random_state
#             background_data = shap.sample(X_train_scaled_df, SHAP_BACKGROUND_SAMPLE_SIZE, random_state=RANDOM_STATE)
#             # Cần truyền hàm predict_proba
#             explainer = shap.KernelExplainer(model.predict_proba, background_data)

#         if explainer is None:
#             print("Error: Failed to initialize SHAP explainer.")
#             return

#         # --- Calculate SHAP values ---
#         print(f"Calculating SHAP values for sample {sample_index}...")
#         # Luôn tính SHAP cho target_instance (1 mẫu)
#         shap_values_output = explainer(target_instance) 
#         # if shap_values_output.values.shape[-1] > 2:
#             # Lấy Explanation slice cho class 1 -> đây là object hoàn chỉnh cho waterfall
#         shap_exp_for_class1 = shap_values_output[0,:] # Index 0 vì chỉ có 1 mẫu, : là all features
#                 # print("DEBUG: Successfully extracted Explanation slice for class 1.")


#         print("SHAP values extraction complete.")

#         shap_exp_for_class1.data = target_instance_original.values[0]

#         # --- Plotting ---
#         # Lấy xác suất dự đoán cho mẫu cụ thể
#         sepsis_probability = model.predict_proba(target_instance)[0, 1]
#         print(f"Sample {sample_index} Predicted Sepsis Probability: {sepsis_probability:.4f}")

#         plt.figure()
#         print("Generating SHAP waterfall plot...")
#         # Bây giờ shap_exp_for_class1 luôn là Explanation object (hoặc đã raise lỗi)
#         shap.plots.waterfall(shap_exp_for_class1, max_display=15, show=False)
#         plt.title(f"SHAP Explanation for Sepsis (Class 1) - {model_type} - Sample {sample_index}\nPredicted Probability: {sepsis_probability:.4f}")
#         # Đảm bảo đường dẫn lưu file tồn tại hoặc dùng đường dẫn tương đối
#         # pdf_filename = f'shap_waterfall_{model_type}_sample_{sample_index}_class_1.pdf' # Lưu vào thư mục hiện tại
#         pdf_filename = f'./shap_waterfall_{model_type}_sample_{sample_index}_class_1.pdf' # Dùng đường dẫn tuyệt đối (cẩn thận)
#         try:
#              plt.savefig(pdf_filename, bbox_inches='tight')
#              print(f"SHAP waterfall plot saved to {pdf_filename}")
#         except Exception as save_err:
#              print(f"Error saving SHAP plot to {pdf_filename}: {save_err}")
#         plt.show()

#     except ImportError:
#         print("SHAP library not found. Please install it: pip install shap")
#     except Exception as e:
#         print(f"An unexpected error occurred during SHAP explanation for model {model_type}: {e}")
#         traceback.print_exc() # In đầy đủ traceback để debug
def explain_with_shap(model, X_train_scaled_df, X_test_scaled_df, X_test_df, predictors, sample_index):
    """
    Giải thích dự đoán cho một mẫu cụ thể bằng SHAP.
    Tự động chọn explainer, lấy SHAP values cho class 1 và vẽ waterfall plot.
    """
    print(f"\n--- SHAP Explanation for Sample Index: {sample_index} ---")

    # --- Input Validation ---
    # (Giữ nguyên các kiểm tra ban đầu)
    if not isinstance(X_test_scaled_df, pd.DataFrame): print("Error: X_test_scaled_df must be a pandas DataFrame."); return
    if not isinstance(X_train_scaled_df, pd.DataFrame): print("Error: X_train_scaled_df must be a pandas DataFrame."); return
    if not isinstance(X_test_df, pd.DataFrame): print("Error: X_test_df must be a pandas DataFrame."); return
    if sample_index < 0 or sample_index >= len(X_test_scaled_df): print(f"Error: Sample index {sample_index} out of bounds."); return
    if not hasattr(model, 'predict_proba'): print(f"Error: Model needs 'predict_proba'."); return
    # Kiểm tra predictors có trong các DataFrame không
    if not all(p in X_train_scaled_df.columns for p in predictors): print("Error: Not all predictors in X_train_scaled_df columns."); return
    if not all(p in X_test_scaled_df.columns for p in predictors): print("Error: Not all predictors in X_test_scaled_df columns."); return
    if not all(p in X_test_df.columns for p in predictors): print("Error: Not all predictors in X_test_df columns."); return

    # --- Chuẩn bị dữ liệu ---
    # Lọc các DataFrame chỉ chứa các cột predictors để đảm bảo tính nhất quán
    X_train_scaled_filtered = X_train_scaled_df[predictors]
    X_test_scaled_filtered = X_test_scaled_df[predictors]
    X_test_original_filtered = X_test_df[predictors]

    target_instance_scaled = X_test_scaled_filtered.iloc[[sample_index]]
    target_instance_original_vals = X_test_original_filtered.iloc[sample_index].values # Lấy giá trị gốc dạng array

    model_type = type(model).__name__
    is_tree_model = model_type in ['RandomForestClassifier', 'GradientBoostingClassifier',
                                   'XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier',
                                   'DecisionTreeClassifier', 'ExtraTreesClassifier']

    try:
        explainer = None
        print(f"Initializing SHAP Explainer ({'Tree' if is_tree_model else 'Kernel'})...")

        # --- Initialize Explainer ---
        if is_tree_model:
            try:
                # Ưu tiên dùng masker (Independent giả định độc lập, Partition/Impute phức tạp hơn)
                masker = shap.maskers.Independent(X_train_scaled_filtered, max_samples=SHAP_BACKGROUND_SAMPLE_SIZE)
                # feature_perturbation='interventional' thường được khuyến nghị hơn 'tree_path_dependent'
                explainer = shap.TreeExplainer(model, masker, feature_perturbation="interventional")
                print("Initialized TreeExplainer with Independent masker.")
            except Exception as masker_err:
                 print(f"Fallback: Initializing TreeExplainer directly with data (masker failed: {masker_err}).")
                 # Sample background data for fallback as well
                 background_data = shap.sample(X_train_scaled_filtered, SHAP_BACKGROUND_SAMPLE_SIZE, random_state=RANDOM_STATE)
                 try:
                     explainer = shap.TreeExplainer(model, background_data)
                 except Exception as tree_err:
                     print(f"Error: Failed to initialize TreeExplainer: {tree_err}")
                     return
        else:
            # Dùng KernelExplainer
            print(f"Sampling {SHAP_BACKGROUND_SAMPLE_SIZE} background samples for KernelExplainer...")
            background_data = shap.sample(X_train_scaled_filtered, SHAP_BACKGROUND_SAMPLE_SIZE, random_state=RANDOM_STATE)
            # KernelExplainer cần hàm trả về xác suất
            explainer = shap.KernelExplainer(model.predict_proba, background_data)
            print("Initialized KernelExplainer.")

        if explainer is None:
            print("Error: Failed to initialize SHAP explainer.")
            return

        # --- Calculate SHAP values using explainer(instance) ---
        print(f"Calculating SHAP values for sample {sample_index}...")
        # explainer(instance) thường trả về Explanation object
        shap_values_output = explainer(target_instance_scaled)

        # --- Extract SHAP explanation for Class 1 ---
        shap_exp_for_class1 = None
        # Kiểm tra cấu trúc của output để lấy đúng class 1
        if isinstance(shap_values_output, shap.Explanation):
            print(f"DEBUG: shap_values_output.values shape: {getattr(shap_values_output, 'values', 'N/A').shape}")
            # Kiểm tra xem có phải output cho nhiều class không
            if hasattr(shap_values_output, 'values') and isinstance(shap_values_output.values, np.ndarray) and shap_values_output.values.ndim == 3:
                # Shape (n_samples, n_features, n_classes), ví dụ (1, n_features, 2)
                if shap_values_output.values.shape[0] == 1 and shap_values_output.values.shape[2] >= 2:
                    # Lấy slice cho mẫu 0, tất cả features, class 1 (index 1)
                    shap_exp_for_class1 = shap_values_output[0, :, 1]
                    print("DEBUG: Extracted Explanation slice for class 1 using [0, :, 1].")
                else:
                     print(f"Error: Unexpected shape for multi-class Explanation values: {shap_values_output.values.shape}")
                     return
            elif hasattr(shap_values_output, 'values') and isinstance(shap_values_output.values, np.ndarray) and shap_values_output.values.ndim == 2:
                 # Shape (n_samples, n_features), ví dụ (1, n_features)
                 # Có thể xảy ra nếu explainer chỉ trả về cho 1 class (cần xác nhận là class nào)
                 # Hoặc nếu dùng KernelExplainer và .shap_values() thì cần tạo Explanation thủ công (đã làm ở trên)
                 # Giả sử trường hợp này là class 1 (cần kiểm tra)
                 print("Warning: SHAP values shape is (samples, features). Assuming class 1 explanation.")
                 shap_exp_for_class1 = shap_values_output[0] # Lấy slice cho mẫu 0
                 print("DEBUG: Sliced Explanation object using [0]. Assuming class 1.")
            else:
                print(f"Error: Unexpected structure or type for shap_values_output.values: {type(getattr(shap_values_output, 'values', None))}")
                return
        else:
             # Xử lý trường hợp explainer trả về kiểu khác (ví dụ: list từ KernelExplainer.shap_values cũ)
             # Phần này phức tạp hơn, phiên bản trên cố gắng dùng explainer() là chủ yếu.
             # Nếu bạn gặp trường hợp này, bạn cần thêm logic xử lý list như mô tả ở điểm 1.
             print(f"Error: SHAP output is not an Explanation object (type: {type(shap_values_output)}). Cannot reliably extract class 1.")
             # Có thể thử logic fallback ở đây nếu cần
             return


        if shap_exp_for_class1 is None:
             print("Error: Could not derive SHAP explanation specifically for class 1.")
             return

        # --- Thay thế dữ liệu hiển thị bằng giá trị gốc ---
        # Đối tượng shap_exp_for_class1 bây giờ là Explanation cho 1 mẫu, 1 class
        try:
            shap_exp_for_class1.data = target_instance_original_vals
            print("DEBUG: Replaced data in Explanation object with original feature values.")
        except Exception as data_err:
            print(f"Warning: Could not set original data in Explanation object: {data_err}. Plot might show scaled values.")

        print("SHAP values extraction for class 1 complete.")

        # --- Plotting ---
        sepsis_probability = model.predict_proba(target_instance_scaled)[0, 1]
        print(f"Sample {sample_index} Predicted Sepsis Probability (Class 1): {sepsis_probability:.4f}")

        plt.figure(figsize=(10, 8)) # Kích thước biểu đồ có thể điều chỉnh
        print("Generating SHAP waterfall plot for Class 1...")
        try:
            # Waterfall plot cần Explanation object cho 1 mẫu, 1 class
            shap.plots.waterfall(shap_exp_for_class1, max_display=15, show=False)
            plt.title(f"SHAP Explanation for Sepsis (Class 1) - {model_type} - Sample {sample_index}\nPredicted Probability: {sepsis_probability:.4f}")
            plt.tight_layout() # Tự động điều chỉnh layout

            pdf_filename = f'./shap_waterfall_{model_type}_sample_{sample_index}_class_1.pdf'
            try:
                 plt.savefig(pdf_filename, bbox_inches='tight')
                 print(f"SHAP waterfall plot saved to {pdf_filename}")
            except Exception as save_err:
                 print(f"Error saving SHAP plot to {pdf_filename}: {save_err}")

            plt.show()
        except Exception as plot_err:
             print(f"Error generating SHAP waterfall plot: {plot_err}")
             traceback.print_exc()
             plt.close() # Đóng figure nếu có lỗi

    except ImportError:
        print("SHAP library not found. Please install it: pip install shap")
    except Exception as e:
        print(f"An unexpected error occurred during SHAP explanation for model {model_type}: {e}")
        traceback.print_exc()

# --- Decision Curve Analysis Functions ---
# (Giữ nguyên các hàm net_benefit và decision_curve_analysis)
def net_benefit(tp, fp, n, threshold):
    """Tính Net Benefit tại một threshold."""
    if threshold >= 1.0 or threshold <= 0.0: return -np.inf # Tránh chia cho 0 hoặc log(0)
    benefit = tp / n
    harm = fp / n * (threshold / (1 - threshold))
    return benefit - harm

def decision_curve_analysis(y_true, y_prob, thresholds):
    """Thực hiện tính toán cho Decision Curve Analysis."""
    n = len(y_true)
    net_benefits = []
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    for threshold in thresholds:
        # Chỉ tính toán nếu threshold hợp lệ (0 < threshold < 1)
        if threshold <= 0 or threshold >= 1:
            net_benefits.append(-np.inf) # Hoặc np.nan tùy cách xử lý
            continue

        # Dự đoán positive nếu xác suất >= threshold
        y_pred_threshold = (y_prob >= threshold).astype(int)

        # Tính TP, FP (chỉ cần 2 cái này và n)
        tp = np.sum((y_pred_threshold == 1) & (y_true == 1))
        fp = np.sum((y_pred_threshold == 1) & (y_true == 0))

        # Tính Net Benefit
        nb = net_benefit(tp, fp, n, threshold)
        net_benefits.append(nb)

    return net_benefits

def run_decision_curve_analysis(models_dict, X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Chạy DCA cho tất cả các mô hình và vẽ biểu đồ.
    """
    print("\n--- Running Decision Curve Analysis (DCA) ---")
    # Bỏ các điểm đầu cuối (0, 1) vì Net Benefit không xác định rõ ở đó
    thresholds = np.linspace(0.01, 0.99, 99)
    dca_results = {}
    trained_models_for_dca = {} # Lưu các model đã được huấn luyện

    for name, model_instance in models_dict.items():
        print(f"Training and getting probabilities for DCA: {name}")
        # Clone model để tránh thay đổi model gốc (đặc biệt quan trọng nếu dùng lại model đã train trước đó)
        from sklearn.base import clone
        model = clone(model_instance)
        try:
            model.fit(X_train_scaled, y_train)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            nb = decision_curve_analysis(y_test, y_pred_proba, thresholds)
            # Chỉ thêm vào kết quả nếu tính toán thành công (nb không phải toàn nan/inf)
            if not np.all(np.isinf(nb)) and not np.all(np.isnan(nb)):
                 dca_results[name] = nb
                 trained_models_for_dca[name] = model
            else:
                 print(f"Warning: Net benefit calculation failed for {name}. Skipping in DCA plot.")

        except AttributeError:
            print(f"Warning: Cannot get probabilities for {name}. Skipping in DCA.")
        except Exception as e:
             print(f"Error processing {name} for DCA: {e}")

    if not dca_results:
        print("Error: No models could be processed for DCA.")
        return None

    # Chiến lược "Treat None" và "Treat All"
    treat_none = np.zeros_like(thresholds)
    prevalence = np.mean(y_test)
    treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    # Calculate the maximum net benefit to set y-axis limits
    max_net_benefit = max([max(nb) for nb in dca_results.values()] + [max(treat_all)])

    # Vẽ biểu đồ DCA
    plt.figure(figsize=(12, 8))
    colors = ['purple', 'green', 'red', 'orange', 'hotpink', 'blue', 'brown']

    for i, (name, nb) in enumerate(dca_results.items()):
        plt.plot(thresholds, nb, label=name, color=colors[i], linewidth=2)

    plt.plot(thresholds, treat_none, 'k--', label='Treat None', linewidth=2)
    plt.plot(thresholds, treat_all, 'k:', label='Treat All', linewidth=2)

    plt.ylim(-0.035, max_net_benefit + 0.05)
    plt.xlim(0, 1)
    plt.xlabel('Threshold Probability', fontsize=12)
    plt.ylabel('Net Benefit', fontsize=12)
    plt.title('Decision Curve Analysis', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("C:/Users/LENOVO/Downloads/dca_curve_comparison.pdf", bbox_inches='tight')
    print("\nDCA plot saved to dca_curve_comparison.pdf")
    plt.show()

    return trained_models_for_dca


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load and Preprocess Data
    preprocess_result = load_and_preprocess_data(
        FILE_PATH, TARGET_COLUMN, COLUMNS_TO_DROP, TEST_SIZE, RANDOM_STATE
    )

    if preprocess_result:
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, predictors, scaler = preprocess_result
    else:
        print("Exiting due to data loading/preprocessing errors.")
        exit(1) # Thoát với mã lỗi

    # 2. Define Models
    models_dict = define_models(RANDOM_STATE)

    # 3. Select ONE model for detailed evaluation and explanation via user input
    print("\n--- Model Selection for Detailed Analysis (LIME/SHAP) ---")
    model_names = list(models_dict.keys())
    print("Available models:")
    for i, name in enumerate(model_names):
        print(f"{i + 1}: {name}")

    chosen_model_name = None
    chosen_model_instance = None
    while chosen_model_name is None:
        try:
            choice = input(f"Enter the number (1-{len(model_names)}) of the model you want to analyze in detail: ")
            chosen_index = int(choice) - 1 # Convert to 0-based index
            if 0 <= chosen_index < len(model_names):
                chosen_model_name = model_names[chosen_index]
                chosen_model_instance = models_dict[chosen_model_name]
                print(f"You selected: {chosen_model_name}")
                # Không break vội, để lấy instance rồi mới break
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(model_names)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nExecution interrupted by user.")
            exit()
        except Exception as e:
            print(f"An unexpected error occurred during selection: {e}")

    # 4. Train and Evaluate the Chosen Model
    # Kiểm tra xem chosen_model_instance đã được gán chưa (phòng trường hợp lỗi lạ)
    if chosen_model_instance is None:
         print("Error: Could not select model instance. Exiting.")
         exit(1)

    # Huấn luyện và đánh giá chỉ mô hình đã chọn
    trained_model, metrics, y_scores = train_and_evaluate_model(
        chosen_model_name,
        chosen_model_instance, # Truyền instance đã lấy từ dict
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test
    )

    # Chỉ tiếp tục giải thích nếu huấn luyện và đánh giá thành công
    if trained_model and metrics:
        # 5. Explain the Chosen Model with LIME
        explain_with_lime(
            trained_model,
            scaler, # Cần scaler để hàm predict_proba_lime hoạt động
            X_train, # Dữ liệu gốc
            X_test,  # Dữ liệu gốc
            predictors,
            EXPLAIN_SAMPLE_INDEX
        )

        # 6. Explain the Chosen Model with SHAP
        explain_with_shap(
            trained_model,
            X_train_scaled, # SHAP dùng dữ liệu đã scale
            X_test_scaled,
            X_test,
            predictors,
            EXPLAIN_SAMPLE_INDEX
        )
    else:
        print(f"\nSkipping LIME and SHAP explanations due to issues in training/evaluating {chosen_model_name}.")


    # 7. Run Decision Curve Analysis for ALL models
    #    (Hàm này sẽ tự huấn luyện lại các model bên trong)
    # trained_dca_models = run_decision_curve_analysis(
    #     models_dict, # Truyền dict gốc với tất cả model instances
    #     X_train_scaled,
    #     y_train,
    #     X_test_scaled,
    #     y_test
    # )

    print("\n--- Script Execution Finished ---")