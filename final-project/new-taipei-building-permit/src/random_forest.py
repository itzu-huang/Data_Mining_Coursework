# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 14:17:38 2026

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

'''=============25.編碼==============='''
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

'''=========27. 切分 80% 訓練集、20% 測試集============'''
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

'''============25.執行 One-Hot Encoding=================='''
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

X_train_RF = X_train_encoded.copy()
X_test_RF = X_test_encoded.copy()

print("X_train_RF shape =", X_train_RF.shape)
print("X_test_RF shape =", X_test_RF.shape)

'''============27. 基本 Random Forest=================='''
from sklearn.ensemble import RandomForestClassifier
RF_basic = RandomForestClassifier(n_estimators=200,max_depth=8,
                                  random_state=20260609,n_jobs=-1)
RF_basic.fit(X_train_encoded,y_train)

print("Random Forest training accuracy =",RF_basic.score(X_train_encoded, y_train))
print("Random Forest testing accuracy =",RF_basic.score(X_test_encoded, y_test))

'''=========28. Random Forest Parameter Comparison==========='''
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

rf_results = []
rf_models = {}
# 四種參數設定
rf_parameter_sets = [{"Model": "RF1","n_estimators": 100,
                      "max_depth": 6,"min_samples_leaf": 1},
                     {"Model": "RF2","n_estimators": 200,
                      "max_depth": 8,"min_samples_leaf": 1},
                     {"Model": "RF3","n_estimators": 300,
                      "max_depth": 10,"min_samples_leaf": 2},
                     {"Model": "RF4","n_estimators": 500,
                      "max_depth": None,"min_samples_leaf": 2}]

for parameter in rf_parameter_sets:
    RF_model = RandomForestClassifier(n_estimators=parameter["n_estimators"],
                                      max_depth=parameter["max_depth"],
                                      min_samples_leaf=parameter["min_samples_leaf"],
                                      random_state=20260609,n_jobs=-1)
    RF_model.fit(X_train_encoded,y_train)

    train_acc = RF_model.score(X_train_encoded,y_train)
    test_acc = RF_model.score(X_test_encoded,y_test)

    # 儲存模型，後面 Voting 可以使用
    rf_models[parameter["Model"]] = RF_model
    rf_results.append({"Model": parameter["Model"],
                       "n_estimators": parameter["n_estimators"],
                       "max_depth": parameter["max_depth"],
                       "min_samples_leaf": parameter["min_samples_leaf"],
                       "Training_Accuracy": train_acc,
                       "Testing_Accuracy": test_acc,
                       "Accuracy_Gap": train_acc - test_acc})

rf_results_df = pd.DataFrame(rf_results)
rf_results_df = rf_results_df.sort_values(by=["Testing_Accuracy","Accuracy_Gap"],
                                          ascending=[False,True])
print("Random Forest comparison:")
print(rf_results_df)

'''=========29. Choose the Best Random Forest==========='''
best_rf_row = rf_results_df.iloc[0]
best_rf_name = best_rf_row["Model"]
best_RF = rf_models[best_rf_name]

print("Best Random Forest =", best_rf_name)
print("Best n_estimators =",int(best_rf_row["n_estimators"]))
print("Best max_depth =",best_rf_row["max_depth"])
print("Best min_samples_leaf =",int(best_rf_row["min_samples_leaf"]))
print("Best training accuracy =",best_rf_row["Training_Accuracy"])
print("Best testing accuracy =",best_rf_row["Testing_Accuracy"])
