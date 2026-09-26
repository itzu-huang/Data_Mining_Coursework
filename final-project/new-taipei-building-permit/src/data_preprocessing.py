# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 13:23:23 2026

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

'''=========='''
#!!!先做資料分割 (Train-test split)，
#再處理缺失值（特別是使用平均數或中位數進行填補時）
'''=======建立 X 與 y========='''
X_mv=df.drop(columns=["whether_for_public"])
y_mv=df["whether_for_public"]

'''=======切分訓練集與測試集========='''
from sklearn.model_selection import train_test_split
X_train_mv, X_test_mv, y_train_mv, y_test_mv = train_test_split(X_mv,y_mv,test_size=0.2,
                                                    random_state=20260609)

# 建立副本，避免 SettingWithCopyWarning
X_train_mv = X_train_mv.copy()
X_test_mv = X_test_mv.copy()

'''=======處理數值型缺失值(中位數)========='''
numeric_cols=(X_train_mv.select_dtypes(include=["int64", "float64"]).columns)

for col in numeric_cols:
    train_median = X_train_mv[col].median()
    # 訓練資料用訓練集的中位數
    X_train_mv[col] = (X_train_mv[col].fillna(train_median))
    # 測試資料也只能使用訓練集的中位數
    X_test_mv[col] = (X_test_mv[col].fillna(train_median))

'''=======處理類別型缺失值(眾數)========='''
category_cols = (X_train_mv.select_dtypes(include=["object"]).columns)

for col in category_cols:
    train_mode = X_train_mv[col].mode()[0]
    # 訓練資料使用訓練集眾數
    X_train_mv[col] = (X_train_mv[col].fillna(train_mode))
    # 測試資料也使用訓練集眾數
    X_test_mv[col] = (X_test_mv[col].fillna(train_mode))

'''====在檢查一次缺失值===='''
print("X_train missing values =",X_train_mv.isnull().sum().sum())
print("X_test missing values =",X_test_mv.isnull().sum().sum())

print("X_train shape =", X_train_mv.shape)
print("X_test shape =", X_test_mv.shape)
