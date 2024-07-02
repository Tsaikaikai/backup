import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, hinge_loss
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 讀取數據
def read_csv_files(folder_path):
    data_list = []
    labels_list = []
    class_names = []
    
    for i, filename in enumerate(sorted(os.listdir(folder_path))):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            data_list.append(df[['X', 'Y', 'Z']])
            labels_list.append(np.full(len(df), i))
            # 從文件名中提取類別名稱（去掉.csv後綴）
            class_names.append(os.path.splitext(filename)[0])
    
    return np.vstack(data_list), np.concatenate(labels_list), class_names

# 設定數據文件夾路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
train_folder = os.path.join(current_dir, 'train_data')
test_folder = os.path.join(current_dir, 'test_data')

# 讀取訓練數據
X_train, y_train, class_names = read_csv_files(train_folder)

# 讀取測試數據
X_test, y_test, _ = read_csv_files(test_folder)

# 分割訓練數據為訓練集(90%)和驗證集(10%)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# 標準化特徵
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 創建SVM分類器
svm_classifier = svm.SVC(kernel='rbf', random_state=42, decision_function_shape='ovr')

# 訓練模型
svm_classifier.fit(X_train_scaled, y_train)

# 在驗證集上進行預測
y_val_pred = svm_classifier.predict(X_val_scaled)

# 計算驗證集的準確度
val_accuracy = accuracy_score(y_val, y_val_pred)

# 計算驗證集的hinge loss
decision_values = svm_classifier.decision_function(X_val_scaled)
val_loss = hinge_loss(y_val, decision_values)

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Hinge Loss: {val_loss:.4f}")

# 在測試集上進行預測
y_test_pred = svm_classifier.predict(X_test_scaled)

# 創建 3D 散點圖
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# 為每個類別選擇不同的顏色
colors = plt.cm.rainbow(np.linspace(0, 1, len(class_names)))
for i, color in enumerate(colors):
    mask = y_test_pred == i
    ax.scatter(X_test[mask, 0], X_test[mask, 1], X_test[mask, 2], c=[color], label=class_names[i])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title(f'Prediction Results on Test Data\nValidation Accuracy: {val_accuracy:.4f}, Hinge Loss: {val_loss:.4f}')

# 調整布局以確保圖例完全顯示
plt.tight_layout()

# 保存圖片為向量圖格式 (SVG)
output_svg = os.path.join(current_dir, 'prediction_results.svg')
plt.savefig(output_svg, format='svg', dpi=300, bbox_inches='tight')
plt.close()

print(f"Prediction results have been saved as a vector graph: '{output_svg}'")

# 保存 CSV 文件
results_df = pd.DataFrame({
    'X': X_test[:, 0],
    'Y': X_test[:, 1],
    'Z': X_test[:, 2],
    'Predicted_Class': [class_names[i] for i in y_test_pred]
})
output_csv = os.path.join(current_dir, 'prediction_results.csv')
results_df.to_csv(output_csv, index=False)
print(f"Prediction results have also been saved to '{output_csv}'")