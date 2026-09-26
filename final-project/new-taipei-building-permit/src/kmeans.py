# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 21:51:48 2026

@author: Itzu Huang
"""
import pandas as pd
df=pd.read_csv("新北市建築執照存查.csv")
df.info()

target_map={"供公眾": "Public","Y": "Public",
              "非公眾": "Non-public","N": "Non-public"}

df["whether_for_public"]=(df["whether_for_public"]
                            .astype(str)
                            .str.strip()
                            .map(target_map))

# 刪除無法判定目標類別的資料
df=df.dropna(subset=["whether_for_public"]).copy()
print(df["whether_for_public"].value_counts())
#刪除識別型欄位
drop_cols=["license_number",                # 執照號碼
             "proprietor",                  # 起造人
             "building_site",               # 建築地點，內容過度細碎
             "house_address",               # 門牌地址
             "designer",                    # 設計人
             "supervisor",                  # 監造人
             "constructor",                 # 承造人
             "original_construction_lic",   # 原執照字號
             "registration_number"]         # 掛號號碼

df=df.drop(columns=drop_cols)

'''=======清理無效格式========='''
import numpy as np
# 將空白字串轉成真正的缺失值 NaN
df = df.replace(r"^\s*$", np.nan, regex=True)

'''=======建立土地使用分區的衍生變數========='''
# 原始 land_use_zoning 有 500 多種不同寫法，
# 因此依照內容關鍵字整理成較大的類別
def group_land_use(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    x = x.replace("，", ",").rstrip(",")

    if "住宅" in x:
        return "Residential"
    elif ("商業" in x) or ("市場" in x):
        return "Commercial"
    elif ("工業" in x) or ("產業" in x) or ("倉儲" in x):
        return "Industrial"
    elif ("農業" in x) or ("農牧" in x) or ("農舍" in x):
        return "Agricultural"
    elif any(word in x for word in ["學校", "機關", "公園", "道路", "交通",
                                    "停車場", "醫療", "社教", "文教",
                                    "公共設施", "河川", "綠地", "廣場"]):
        return "PublicFacility"
    elif any(word in x for word in ["保育", "保護", "森林",
                                    "水源", "國家公園"]):
        return "Conservation"
    elif any(word in x for word in ["鄉村", "甲種建築", "乙種建築",
                                    "丙種建築", "丁種建築"]):
        return "RuralConstruction"
    else:
        return "Other"

df["land_use_group"] = df["land_use_zoning"].apply(group_land_use)

print("土地使用分區整理結果：")
print(df["land_use_group"].value_counts(dropna=False))

'''=======建立建築物用途的衍生變數========='''
# use_of_buildings 原始文字格式非常不一致，
# 例如：集合住宅、H2集合住宅、H2-集合住宅
# 因此依用途關鍵字建立較大的類別
def group_building_use(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if any(word in x for word in ["集合住宅", "住宅", "農舍", "宿舍"]):
        return "Residential"
    elif any(word in x for word in ["工廠", "廠房", "倉庫", "作業廠房"]):
        return "Industrial"
    elif any(word in x for word in [
        "店舖", "店鋪", "商場", "辦公",
        "餐廳", "旅館", "市場", "金融", "商業"]):
        return "Commercial"
    elif any(word in x for word in [
        "學校", "教室", "幼兒園", "醫院",
        "診所", "社會福利", "圖書館",
        "活動中心", "寺廟", "教會"]):
        return "PublicInstitution"
    elif ("停車" in x) or ("車庫" in x):
        return "Parking"
    elif any(word in x for word in [
        "屋頂", "機房", "水塔",
        "電信", "防空避難"]):
        return "AuxiliaryFacility"
    else:
        return "Other"


df["building_use_group"]=(
    df["use_of_buildings"].apply(group_building_use))

print("建築物用途整理結果：")
print(df["building_use_group"].value_counts(dropna=False))

'''=======數值欄位格式轉換========='''
numeric_cols = ["number_of_stories","ground_floor","households",
                "building_height","total_floor_area","project_cost",
                "statutory_open_space","statutory_number_of_vehic",
                "vehicles_parked_award","vehicles_parked_own"]

for col in numeric_cols:
    # 無法轉成數字的內容會變成 NaN
    df[col] = pd.to_numeric(df[col],errors="coerce")
    # 層數、面積、造價和停車位不應為負數
    df.loc[df[col] < 0, col] = np.nan

'''=======清理 building_area========='''
import re
def clean_building_area(x):
    if pd.isna(x):
        return np.nan
    x = str(x).replace(",", "").strip()
    # 若本身就是單一數值，直接轉換
    try:
        return float(x)
    except ValueError:
        # 擷取字串中的所有數字
        numbers = re.findall(r"\d+(?:\.\d+)?",x)
        if len(numbers) == 0:
            return np.nan
        # 若包含騎樓、其他等多個面積，將面積加總
        return sum(float(number) for number in numbers)

df["building_area"]=(df["building_area"].apply(clean_building_area))

'''=======清理建蔽率與容積率========='''
def clean_ratio(x, max_value=None):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    invalid_values = ["","%","％","#######%","#######％","nan"]
    if x in invalid_values:
        return np.nan
    # 判斷原始資料是否使用百分比
    contains_percent = ("%" in x) or ("％" in x)
    x = (x.replace("%", "").replace("％", "").replace(",", ""))
    value = pd.to_numeric(x,errors="coerce")
    if pd.isna(value):
        return np.nan
    # 若原本有百分比符號，要除以 100
    if contains_percent:
        value = value / 100
    if value < 0:
        return np.nan
    # 超出合理範圍時轉成缺失值
    if max_value is not None and value > max_value:
        return np.nan
    return float(value)

# 建蔽率合理範圍設定為 0 到 1
df["building_coverage_ratio"] = (df["building_coverage_ratio"]
                                 .apply(lambda x: clean_ratio(x, max_value=1)))

# 容積率可能大於 1，例如 300% = 3
df["volume_rate"] = (df["volume_rate"]
                     .apply(lambda x: clean_ratio(x, max_value=10)))

'''=======民國日期轉換========='''
def convert_roc_date(x):
    if pd.isna(x):
        return pd.NaT
    x = str(x).strip()
    if x in ["", "//", "nan"]:
        return pd.NaT
    # 情況一：1130207，代表民國113年02月07日
    if re.fullmatch(r"\d{7}", x):
        roc_year = int(x[0:3])
        month = int(x[3:5])
        day = int(x[5:7])
    # 情況二：113/02/07
    else:
        result = re.fullmatch(r"(\d{2,3})/(\d{1,2})/(\d{1,2})",x)
        if result is None:
            return pd.NaT
        roc_year = int(result.group(1))
        month = int(result.group(2))
        day = int(result.group(3))
    # 排除明顯不合理的民國年份
    if roc_year < 1 or roc_year > 200:
        return pd.NaT
    western_year = roc_year + 1911
    try:
        return pd.Timestamp(year=western_year,month=month,day=day)
    except ValueError:
        return pd.NaT

date_cols = ["date_licensing","date_the_permit",
             "commencement_date","completion_date"]

for col in date_cols:
    df[col + "_clean"]=(df[col].apply(convert_roc_date))

'''=======建立日期衍生變數========='''
# 發照年度
df["licensing_year"]=(df["date_licensing_clean"].dt.year)
# 發照月份
df["licensing_month"]=(df["date_licensing_clean"].dt.month)
# 領照日期減發照日期
df["permit_wait_days"]=(df["date_the_permit_clean"]-
                        df["date_licensing_clean"]).dt.days
# 負數代表日期順序不合理，改為缺失值
df.loc[df["permit_wait_days"] < 0,"permit_wait_days"] = np.nan
# 竣工日期減開工日期
df["construction_duration_days"] = (df["completion_date_clean"]
                                    -df["commencement_date_clean"]).dt.days
# 負數不具有合理意義，改為缺失值
df.loc[df["construction_duration_days"] < 0,
       "construction_duration_days"]=np.nan

'''=======建立總停車位========='''
parking_cols = ["statutory_number_of_vehic","vehicles_parked_award",
                "vehicles_parked_own"]
# min_count=3 表示三個停車欄位都有效時才計算
# 若其中任何欄位缺失，總停車位先維持 NaN
df["total_parking_spaces"] = (df[parking_cols].sum(axis=1,min_count=3))

'''=======刪除原始高基數與日期欄位========='''

derived_drop_cols = ["land_use_zoning","use_of_buildings","date_licensing",
                     "date_the_permit","commencement_date","completion_date",
                     "date_licensing_clean","date_the_permit_clean",
                     "commencement_date_clean","completion_date_clean"]
df = df.drop(columns=derived_drop_cols)
print("資料清理後的資料筆數與欄位數：",df.shape)

print("資料清理後的欄位型態：")
df.info()

print("資料清理後的缺失值：",df.isnull().sum().sort_values(ascending=False))

'''========建立 X 與 y==================='''
X = df.drop(columns=["whether_for_public"])
y = df["whether_for_public"]

'''=============編碼==============='''
X = df.drop(columns=["whether_for_public"])
y = df["whether_for_public"]

# 目標變數 Label Encoding
from sklearn.preprocessing import LabelEncoder
le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y)
print("Target classes =", le_y.classes_)

# 找出數值型與類別型欄位
numeric_cols = (X.select_dtypes(include=["number"]).columns.tolist())
category_cols = (X.select_dtypes(include=["object", "category"])
                 .columns.tolist())
print("Numeric columns =", numeric_cols)
print("Categorical columns =", category_cols)

# 先建立 One-Hot Encoder，但此處尚未 fit
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False,handle_unknown="ignore")

'''=========切分 80% 訓練集、20% 測試集============'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y_encoded,test_size=0.2,
                                                    random_state=20260609,
                                                    stratify=y_encoded)
print("X_train shape =", X_train.shape)
print("X_test shape =", X_test.shape)
# 建立副本
X_train = X_train.copy()
X_test = X_test.copy()

print("X_train shape =", X_train.shape)
print("X_test shape =", X_test.shape)
print("Training target distribution:")
print(pd.Series(y_train).value_counts())
print("Testing target distribution:")
print(pd.Series(y_test).value_counts())

'''===========處理缺失值==============='''
from sklearn.impute import SimpleImputer
# ---------- 數值型缺失值 ----------
numeric_imputer = SimpleImputer(strategy="median")

X_train_numeric = numeric_imputer.fit_transform(X_train[numeric_cols])

# 測試資料只能 transform
X_test_numeric = numeric_imputer.transform(X_test[numeric_cols])

X_train_numeric = pd.DataFrame(X_train_numeric,columns=numeric_cols,
                               index=X_train.index)

X_test_numeric = pd.DataFrame(X_test_numeric,columns=numeric_cols,
                              index=X_test.index)

# ---------- 類別型缺失值 ----------
category_imputer = SimpleImputer(strategy="most_frequent")

X_train_category_imputed=(category_imputer.fit_transform(X_train[category_cols]))
# 測試資料只能 transform
X_test_category_imputed = (category_imputer.transform(X_test[category_cols]))

X_train_category_imputed = pd.DataFrame(X_train_category_imputed,
                                        columns=category_cols,
                                        index=X_train.index)

X_test_category_imputed = pd.DataFrame(X_test_category_imputed,
                                       columns=category_cols,
                                       index=X_test.index)

print("Missing values after imputation in training data =",
      X_train_numeric.isnull().sum().sum()
      +X_train_category_imputed.isnull().sum().sum())

print("Missing values after imputation in testing data =",
      X_test_numeric.isnull().sum().sum()
      +X_test_category_imputed.isnull().sum().sum())

'''============執行 One-Hot Encoding=================='''
# 訓練資料 fit + transform
X_train_category_encoded = ohe.fit_transform(X_train_category_imputed)
# 測試資料只能 transform
X_test_category_encoded = ohe.transform(X_test_category_imputed)

encoded_category_names = (ohe.get_feature_names_out(category_cols))

X_train_category_encoded = pd.DataFrame(X_train_category_encoded,
                                        columns=encoded_category_names,
                                        index=X_train.index)

X_test_category_encoded = pd.DataFrame(X_test_category_encoded,
                                       columns=encoded_category_names,
                                       index=X_test.index)
# 合併數值型與類別型資料
X_train_encoded = pd.concat([X_train_numeric,X_train_category_encoded],axis=1)
X_test_encoded = pd.concat([X_test_numeric,X_test_category_encoded],axis=1)

print("Encoded training shape =",X_train_encoded.shape)
print("Encoded testing shape =",X_test_encoded.shape)

'''=============標準化 StandardScaler=================='''
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
# 訓練資料 fit + transform
X_train_std = ss.fit_transform(X_train_encoded)
# 測試資料只能 transform
X_test_std = ss.transform(X_test_encoded)
X_train_std = pd.DataFrame(X_train_std,columns=X_train_encoded.columns,
                           index=X_train_encoded.index)

X_test_std = pd.DataFrame(X_test_std,columns=X_test_encoded.columns,
                          index=X_test_encoded.index)

print("Missing values in X_train_std =",X_train_std.isnull().sum().sum())
print("Missing values in X_test_std =",X_test_std.isnull().sum().sum())

print("X_train_std shape =", X_train_std.shape)
print("X_test_std shape =", X_test_std.shape)

'''=========39. K-Means: Elbow Method, use K = 4==========='''
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

X_kmeans = pd.concat([X_train_std, X_test_std],axis=0)
X_kmeans = X_kmeans.sort_index()

print("K-Means data shape =", X_kmeans.shape)

#Elbow Method
sse_list = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k,random_state=20260609,n_init=10)
    kmeans.fit(X_kmeans)
    sse_list.append(kmeans.inertia_)

# 整理 SSE 結果
elbow_df = pd.DataFrame({"K": list(K_range),"SSE": sse_list})
print("Elbow Method SSE:")
print(elbow_df)

#畫 Elbow Method 圖
plt.figure(figsize=(8, 5))
plt.plot(elbow_df["K"],elbow_df["SSE"],marker="o")

plt.xlabel("Number of Clusters K")
plt.ylabel("SSE")
plt.title("Elbow Method for K-Means")
plt.xticks(list(K_range))
plt.grid(True)
plt.show()

#依照題目規定，今年使用 K = 4
kmeans_4 = KMeans(n_clusters=4,random_state=20260609,n_init=10)
cluster_k4 = kmeans_4.fit_predict(X_kmeans)

print("K-Means with K=4 SSE =", kmeans_4.inertia_)
print("Cluster counts when K=4:")
print(pd.Series(cluster_k4).value_counts().sort_index())

'''=========40. K-Means: Include Target Variable==========='''
from sklearn.preprocessing import StandardScaler
import pandas as pd

X_kmeans_no_target = X_kmeans.copy()
print("K-Means data without target shape =", X_kmeans_no_target.shape)

y_kmeans = pd.Series(y_encoded,index=X.index,name="whether_for_public_encoded")
y_kmeans = y_kmeans.loc[X_kmeans_no_target.index]

#將 target 標準化後加入資料
target_scaler = StandardScaler()
y_kmeans_scaled = target_scaler.fit_transform(y_kmeans.to_frame())

y_kmeans_scaled = pd.DataFrame(y_kmeans_scaled,columns=["whether_for_public_scaled"],
                               index=X_kmeans_no_target.index)

X_kmeans_with_target = pd.concat([X_kmeans_no_target, y_kmeans_scaled],axis=1)
print("K-Means data with target shape =", X_kmeans_with_target.shape)

#加入 target 後的 Elbow Method
sse_list_with_target = []
K_range = range(1, 11)
for k in K_range:
    kmeans_target = KMeans(n_clusters=k,random_state=20260609,n_init=10)
    kmeans_target.fit(X_kmeans_with_target)
    sse_list_with_target.append(kmeans_target.inertia_)

elbow_target_df = pd.DataFrame({"K": list(K_range),"SSE": sse_list_with_target})

print("Elbow Method SSE with target:")
print(elbow_target_df)

plt.figure(figsize=(8, 5))
plt.plot(elbow_target_df["K"],elbow_target_df["SSE"],marker="o")

plt.xlabel("Number of Clusters K")
plt.ylabel("SSE")
plt.title("Elbow Method for K-Means with Target Variable")
plt.xticks(list(K_range))
plt.grid(True)
plt.show()

'''加入目標變數且 K=5 的分群結果'''
K_target = 5
kmeans_target_final = KMeans(n_clusters=K_target,random_state=20260609,n_init=10)
cluster_with_target = kmeans_target_final.fit_predict(X_kmeans_with_target)

print("K-Means with target, K =", K_target)
print("K-Means with target SSE =",kmeans_target_final.inertia_)
print("Cluster counts with target:")
print(pd.Series(cluster_with_target).value_counts().sort_index())

'''=========41. Compare SSE==========='''
sse_no_target = kmeans_4.inertia_
sse_with_target = kmeans_target_final.inertia_

print("K-Means without target, K=4 SSE =", sse_no_target)
print("K-Means with target, K=5 SSE =", sse_with_target)

if sse_with_target < sse_no_target:
    print("The clustering result with target variable has lower SSE.")
else:
    print("The clustering result without target variable has lower SSE.")

'''=========42. Silhouette Coefficient Comparison==========='''
from sklearn.metrics import silhouette_score
import pandas as pd
# 1. 不加入 target，K=4 的 Silhouette
silhouette_no_target = silhouette_score(X_kmeans_no_target,cluster_k4)

print("Silhouette coefficient without target, K=4 =",silhouette_no_target)

# 2. 加入 target，K=5 的 Silhouette
silhouette_with_target = silhouette_score(X_kmeans_with_target,cluster_with_target)

print("Silhouette coefficient with target, K=5 =",silhouette_with_target)

# 3. 整理比較結果
silhouette_results = pd.DataFrame({"Clustering_Result": ["Without target, K=4",
                                                         "With target, K=5"],
                                   "K": [4,K_target],"SSE": [kmeans_4.inertia_,
                                                             kmeans_target_final
                                                             .inertia_],
                                   "Silhouette_Coefficient": [silhouette_no_target,
                                                              silhouette_with_target]})
print("Silhouette comparison:")
print(silhouette_results)

#判斷Silhouette
if silhouette_with_target > silhouette_no_target:
    print("The clustering result with target has a higher silhouette coefficient.")
else:
    print("The clustering result without target has a higher silhouette coefficient.")

'''=========43. K-Means Cluster Profile==========='''
#將 K=4 的分群結果放回原始資料
df_cluster_profile = df.loc[X_kmeans_with_target.index].copy()
df_cluster_profile["Cluster"] = cluster_with_target

print("Cluster counts:")
print(df_cluster_profile["Cluster"].value_counts().sort_index())

# 2. 每群的目標變數分布，注意：target 沒有放進 K-Means，只是用來輔助解釋

target_count_table = pd.crosstab(df_cluster_profile["Cluster"],
                                 df_cluster_profile["whether_for_public"])

target_ratio_table = pd.crosstab(df_cluster_profile["Cluster"],
                                 df_cluster_profile["whether_for_public"],
                                 normalize="index")
print("Target count by cluster:")
print(target_count_table)
print("Target ratio by cluster:")
print(target_ratio_table)

# 3. 每群數值變數平均數
profile_numeric_cols = ["number_of_stories",
                        "ground_floor",
                        "households",
                        "building_height",
                        "total_floor_area",
                        "project_cost",
                        "building_area",
                        "statutory_open_space",
                        "building_coverage_ratio",
                        "volume_rate",
                        "statutory_number_of_vehic",
                        "vehicles_parked_award",
                        "vehicles_parked_own",
                        "total_parking_spaces",
                        "licensing_year",
                        "licensing_month",
                        "permit_wait_days",
                        "construction_duration_days"]

# 只保留資料中實際存在的欄位
profile_numeric_cols = [col for col in profile_numeric_cols
                        if col in df_cluster_profile.columns]

cluster_numeric_profile = (df_cluster_profile.groupby("Cluster")
                           [profile_numeric_cols].mean().round(2))
print("Numeric profile by cluster:")
print(cluster_numeric_profile)

# 4. 每群土地使用分區比例
land_use_profile = pd.crosstab(df_cluster_profile["Cluster"],
                               df_cluster_profile["land_use_group"],
                               normalize="index").round(3)
print("Land use group ratio by cluster:")
print(land_use_profile)

# 5. 每群建築物用途比例
building_use_profile = pd.crosstab(df_cluster_profile["Cluster"],
                                   df_cluster_profile["building_use_group"],
                                   normalize="index").round(3)
print("Building use group ratio by cluster:")
print(building_use_profile)

# 6. 每群主要土地使用分區與主要建築用途
cluster_main_land_use = (df_cluster_profile
                         .groupby("Cluster")["land_use_group"]
                         .agg(lambda x: x.value_counts().idxmax()))
cluster_main_building_use = (df_cluster_profile
                            .groupby("Cluster")["building_use_group"]
                            .agg(lambda x: x.value_counts().idxmax()))


cluster_summary = pd.DataFrame({"Cluster_Size": df_cluster_profile["Cluster"]
                                .value_counts().sort_index(),
                                "Main_Land_Use": cluster_main_land_use,
                                "Main_Building_Use": cluster_main_building_use})
print("Cluster summary:")
print(cluster_summary)

'''=========44. Assign Meaningful Cluster Names==========='''
cluster_name_map = {0: "Agricultural Residential Buildings",
                    1: "Special Residential Buildings",
                    2: "Industrial Buildings",
                    3: "General Residential Buildings",
                    4: "Public Facility and Special-use Buildings"}

df_cluster_profile["Cluster_Name"] = (df_cluster_profile["Cluster"]
                                      .map(cluster_name_map))
print("Cluster name count:")
print(df_cluster_profile["Cluster_Name"].value_counts())

'''=========45. Majority Voting within Each Cluster==========='''
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

df_cluster_vote = df_cluster_profile.copy()
# 查看每個 cluster 中 Public / Non-public 的數量
cluster_target_count = pd.crosstab(df_cluster_vote["Cluster"],
                                   df_cluster_vote["whether_for_public"])

print("Target count by cluster:")
print(cluster_target_count)

# 找出每個 cluster 的多數類別
cluster_majority_class = cluster_target_count.idxmax(axis=1)
print("Majority class in each cluster:")
print(cluster_majority_class)

# 將每筆資料預測為該 cluster 的多數類別
df_cluster_vote["Predicted_Class"] = (df_cluster_vote["Cluster"]
                                      .map(cluster_majority_class))
# 計算分類正確率
cluster_majority_accuracy = accuracy_score(df_cluster_vote["whether_for_public"]
                                           ,df_cluster_vote["Predicted_Class"])
print("Classification accuracy using cluster majority voting =",
      cluster_majority_accuracy)
# 混淆矩陣
print("Confusion matrix:")
print(confusion_matrix(df_cluster_vote["whether_for_public"],
                       df_cluster_vote["Predicted_Class"],
                       labels=["Non-public", "Public"]))
# 分類報告
print("Classification report:")
print(classification_report(df_cluster_vote["whether_for_public"],
                            df_cluster_vote["Predicted_Class"],
                            zero_division=0))
