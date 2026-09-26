# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 20:37:38 2026

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

'''=============Decision Tree (題目 5～7=================='''
X = df.drop(columns=["whether_for_public"])
y = df["whether_for_public"]

#目標變數 Label Encoding
from sklearn.preprocessing import LabelEncoder
# Non-public = 0  Public = 1
le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y)
print("Target classes =", le_y.classes_)

# 第7題：80% training、20% testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y_encoded,test_size=0.2,
                                                    random_state=20260609,
                                                    stratify=y_encoded)
print("X_train shape =", X_train.shape)
print("X_test shape =", X_test.shape)

# 找出數值型與類別型變數
numeric_cols = (X_train.select_dtypes(include=["number"]).columns.tolist())

category_cols = (X_train.select_dtypes(include=["object", "category"])
                 .columns.tolist())

print("Numeric variables =", numeric_cols)
print("Categorical variables =", category_cols)

'''=============6.缺失值處理及One-Hot Encoding============='''
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer(
    transformers=[
        # 數值型缺失值：訓練集中位數
        ("num",SimpleImputer(strategy="median"),numeric_cols),
        # 類別型缺失值：眾數，再進行 One-Hot Encoding
        ("cat",Pipeline(steps=[( "imputer",SimpleImputer(
            strategy="most_frequent")),
            ("onehot",OneHotEncoder(sparse_output=False,
                                    handle_unknown="ignore"))]),
            category_cols)])
# 訓練資料 fit + transform
# 取得編碼後欄位名稱
# 測試資料只能 transform

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()

print("Number of encoded variables =",len(feature_names))

print("Encoded variables:",feature_names)

'''=============7.建立基本 Decision Tree============='''
from sklearn.tree import DecisionTreeClassifier

DT_baseline = DecisionTreeClassifier(random_state=20260609)
DT_baseline.fit(X_train_encoded,y_train)

print("Baseline training accuracy =",DT_baseline.score(X_train_encoded,y_train))
print("Baseline testing accuracy =",DT_baseline.score(X_test_encoded,y_test))

'''==========8.Overfitting Test==================='''
overfitting_results = []
split_values = [round(value, 2)
                for value in np.arange(0.10,0.00,-0.01)]

for split_value in split_values:
    model = DecisionTreeClassifier(criterion="gini",min_samples_leaf=2,
                                   min_samples_split=split_value,
                                   random_state=20260609)
    model.fit(X_train_encoded,y_train)
    train_acc = model.score(X_train_encoded,y_train)
    test_acc = model.score(X_test_encoded,y_test)
    overfitting_results.append({"min_samples_split": split_value,
                                "training_accuracy": train_acc,
                                "testing_accuracy": test_acc,
                                "accuracy_gap": train_acc - test_acc,
                                "tree_depth": model.get_depth(),
                                "number_of_leaves": model.get_n_leaves()})

overfitting_df = pd.DataFrame(overfitting_results)
print("Overfitting results:")
print(overfitting_df)

# 找出測試正確率最高的 min_samples_split
best_index = (overfitting_df["testing_accuracy"].idxmax())
BEST_SPLIT = float(overfitting_df.loc[best_index,"min_samples_split"])

print("Best min_samples_split =",BEST_SPLIT)
print("Best testing accuracy =",overfitting_df.loc[best_index,
                                                   "testing_accuracy"])

# 判斷最佳值之後，較小的 split 是否開始過度擬合
best_train_acc = overfitting_df.loc[best_index,"training_accuracy"]
best_test_acc = overfitting_df.loc[best_index,"testing_accuracy"]
overfit_candidates = overfitting_df[(overfitting_df["min_samples_split"]
                                     <BEST_SPLIT)&
                                    (overfitting_df["training_accuracy"]
                                     >best_train_acc)&
                                    (overfitting_df["testing_accuracy"]
                                     <best_test_acc)].sort_values("min_samples_split",
                                                                  ascending=False)

if len(overfit_candidates)>0:
    overfit_start = (overfit_candidates.iloc[0]["min_samples_split"])
    print("Overfitting begins at min_samples_split =",overfit_start)
else:
    print("No clear overfitting point was found.")

'''=============9.10.Chi-Square & Model Selection============='''
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

minmax = MinMaxScaler()

# 只能使用 training data fit
X_train_chi = minmax.fit_transform(X_train_encoded)
# testing data 只能 transform
X_test_chi = minmax.transform(X_test_encoded)

chi_selector = SelectKBest(score_func=chi2,k=5)
X_train_chi_selected = (chi_selector.fit_transform(X_train_chi,y_train))
X_test_chi_selected = (chi_selector.transform(X_test_chi))

chi_features = feature_names[chi_selector.get_support()]
print("Top 5 Chi-Square variables:")
print(chi_features)

selector_tree = DecisionTreeClassifier(criterion="gini",min_samples_leaf=2,
                                       min_samples_split=BEST_SPLIT,
                                       random_state=20260609)
selector_tree.fit(X_train_encoded,y_train)
feature_importance = (selector_tree.feature_importances_)

model_top_indices = np.argsort(feature_importance)[::-1][:5]
model_features = feature_names[model_top_indices]

print("Top 5 Model Selection variables:")
print(model_features)
print("Feature importance:")
print(feature_importance[model_top_indices])

X_train_model_selected = (X_train_encoded[:,model_top_indices])
X_test_model_selected = (X_test_encoded[:,model_top_indices])

'''========9.10.比較兩種Feature Selection與兩種Criterion=========='''
feature_selection_results = []
feature_datasets = {"Chi-Square":(X_train_chi_selected,X_test_chi_selected),
                    "Model Selection":(X_train_model_selected,
                                       X_test_model_selected)}

for method_name, data in feature_datasets.items():
    selected_train = data[0]
    selected_test = data[1]
    for criterion_name in ["gini","entropy"]:
        model = DecisionTreeClassifier(criterion=criterion_name,
                                       min_samples_leaf=2,
                                       min_samples_split=BEST_SPLIT,
                                       random_state=20260609)
        model.fit(selected_train,y_train)

        feature_selection_results.append({"Feature_Selection": method_name,
                                          "Criterion": criterion_name,
                                          "Training_Accuracy": model.score(
                                              selected_train,y_train),
                                          "Testing_Accuracy": model.score(
                                              selected_test,y_test),
                                          "Depth": model.get_depth(),
                                          "Leaves": model.get_n_leaves()})
feature_selection_df = pd.DataFrame(feature_selection_results)
print("Feature selection comparison:") #比較結果
print(feature_selection_df) # Model Selection + Gini 最好

'''=========11.All Variables vs Selected Variables==========='''
comparison_results = []
comparison_models = {}
comparison_datasets = {"All Variables": (X_train_encoded,X_test_encoded),
                       "Chi-Square Top 5": (X_train_chi_selected,
                                            X_test_chi_selected),
                       "Model Selection Top 5": (X_train_model_selected,
                                                 X_test_model_selected)}

for variable_set, data in comparison_datasets.items():
    selected_train = data[0]
    selected_test = data[1]
    for criterion_name in ["gini","entropy"]:
        model = DecisionTreeClassifier(criterion=criterion_name,
                                       min_samples_leaf=2,
                                       min_samples_split=BEST_SPLIT,
                                       max_depth=8,random_state=20260609)
        model.fit(selected_train,y_train)

        comparison_models[(variable_set,criterion_name)]=model

        comparison_results.append({"Variables": variable_set,
                                   "Criterion": criterion_name,
                                   "Training_Accuracy": model.score(
                                       selected_train,y_train),
                                   "Testing_Accuracy": model.score(
                                       selected_test,y_test),
                                   "Depth": model.get_depth(),
                                   "Leaves": model.get_n_leaves(),
                                   "Nodes": model.tree_.node_count})

comparison_df = pd.DataFrame(comparison_results)
comparison_df = comparison_df.sort_values(by="Testing_Accuracy",
                                          ascending=False)
print("Final Decision Tree comparison:")
print(comparison_df)

'''==========12.Final Decision Tree================'''

final_tree = DecisionTreeClassifier(criterion="gini",min_samples_leaf=2,
                                    min_samples_split=BEST_SPLIT,
                                    max_depth=8,random_state=20260609)
final_tree.fit(X_train_encoded,y_train)

final_train_acc = final_tree.score(X_train_encoded,y_train)

final_test_acc = final_tree.score(X_test_encoded,y_test)

print("Final tree training accuracy =", final_train_acc)
print("Final tree testing accuracy =", final_test_acc)
print("Tree depth =", final_tree.get_depth())
print("Number of leaves =", final_tree.get_n_leaves())
print("Number of nodes =", final_tree.tree_.node_count)

'''===========13.Decision Rules===================='''
from sklearn.tree import export_text
tree_rules = export_text(final_tree,feature_names=list(feature_names),
                         decimals=3)
print(tree_rules)

'''==========14.5-Fold Cross-Validation=============='''
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# 使用完整資料
X_full = df.drop(columns=["whether_for_public"])

y_full = le_y.fit_transform(df["whether_for_public"])

full_numeric_cols = (X_full.select_dtypes(include=["number"]).columns.tolist())

full_category_cols = (X_full.select_dtypes(include=["object", "category"])
                      .columns.tolist())

cv_preprocessor = ColumnTransformer(transformers=[("num",SimpleImputer(
    strategy="median"),
    full_numeric_cols),("cat",Pipeline(steps=[(
        "imputer",
        SimpleImputer(strategy="most_frequent")),
        ("onehot",OneHotEncoder(handle_unknown="ignore"))]),
        full_category_cols)])

cv_tree = DecisionTreeClassifier(criterion="gini",min_samples_leaf=2,
                                 min_samples_split=BEST_SPLIT,max_depth=8,
                                 random_state=20260609)
cv_pipeline = Pipeline(steps=[("preprocessing",cv_preprocessor),
                              ("decision_tree",cv_tree)])
cv_method = StratifiedKFold(n_splits=5,shuffle=True,random_state=20260609)

cv_scores = cross_val_score(cv_pipeline,X_full,y_full,cv=cv_method,
                            scoring="accuracy",n_jobs=1)
print("5-fold CV accuracy =", cv_scores)
print("Mean CV accuracy =",cv_scores.mean())
print("CV standard deviation =",cv_scores.std())
