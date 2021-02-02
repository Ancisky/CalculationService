import time

import pandas as pd
import numpy as np
import random
import os
import sklearn
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor

from compute import file_util, config

'''
    环境设置
'''
# 正常显示中文
from pylab import mpl
# filter warnings
import warnings

warnings.filterwarnings('ignore')
# 正常显示符号
from matplotlib import rcParams

mpl.rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 默认全局随机种子
GLOBAL_SEED = 1
def set_seed(seed):
    '''
        设置全局随机种子
    :param seed: 默认：@GLOBAL_SEED
    '''
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print("设置随机种子：", seed)


# 全局线程ID,用于固定dataloader随机种子
GLOBAL_WORKER_ID = None
def worker_init_fn(worker_id):
    '''
    设置dataloader加载随机种子,固定加载顺序
    :param worker_id:
    :return:
    '''
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)
    print("设置dataloader种子")


# 标准化
GLOBAL_SCALE_X = StandardScaler()
def data_standardization(x, y=None):
    mean_cols = x.mean()
    # x=x.fillna(mean_cols)  #填充缺失值
    # x=pd.get_dummies(x)    #独热编码
    # 归一化
    # mm_x = MinMaxScaler()
    # x = mm_x.fit_transform(x)
    # 标准化
    x = GLOBAL_SCALE_X.fit_transform(x)
    if not y is None:
        # y = np.log(y)  # 平滑处理Y
        y = np.array(y).reshape(-1, 1)
        # 转一维
        y = y.ravel()
    return x, y


# 加载数据
def loadXY(datafilepath):
    label_flag = 'OUT'
    data = pd.read_table(datafilepath, sep=',')
    x = data.loc[:, data.columns != label_flag]
    y = data.loc[:, label_flag]
    return data_standardization(x, y)


def train(prop, k_fold=5, test_size=0.2):
    # 0.settings
    set_seed(GLOBAL_SEED)
    cv = k_fold  # cross-validation generator
    if cv == 1:
        cv = LeaveOneOut()

    # 1.basic learner nets
    knn = KNeighborsRegressor(leaf_size=3, n_neighbors=2, p=1, weights='distance')
    svr = GridSearchCV(SVR(), param_grid={"C": np.logspace(0, 2, 4), "gamma": np.logspace(-2, 2, 7)}, n_jobs=-1)
    ridge = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0))
    mlp = MLPRegressor(hidden_layer_sizes=(50, 100, 50), max_iter=700)
    rf = RandomForestRegressor()
    gbdt = GradientBoostingRegressor()
    # 2.metal model net
    metal_model = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0))
    # 3.stacking model
    stacking_model = StackingRegressor(
        estimators=[('KNN', knn), ('SVR', svr), ('Ridge', ridge), ('MLP', mlp), ('RF', rf), ('GBDT', gbdt)],
        final_estimator=metal_model,
        n_jobs=-1, cv=cv  # cross validation
    )

    # 4.load data
    x, y = loadXY(config.data_load_path[prop])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=True)

    # 5.train model(stacking模型，已经内置交叉验证)
    stacking_model.fit(x_train, y_train)
    # val-scores
    result = cross_validate(stacking_model, x_train, y_train, scoring=['neg_mean_absolute_error','neg_mean_squared_error','r2'], cv=cv)
    mae_val = result['test_neg_mean_absolute_error'].mean()
    mse_val = result['test_neg_mean_squared_error'].mean()
    r2_val = result['test_r2'].mean()
    # test-score
    pred = stacking_model.predict(x_test)
    mae_test = sklearn.metrics.mean_absolute_error(y_test, pred).mean()
    mse_test = sklearn.metrics.mean_squared_error(y_test, pred).mean()
    r2_test = sklearn.metrics.r2_score(y_test, pred).mean()
    # show
    print("验证集: MAE:%f, MSE:%f, R2:%f\n"
          "测试集: MAE:%f, MSE:%f, R2:%f"
          % (mae_val, mse_val, r2_val,
             mae_test, mse_test, r2_test))

    # 7.save model
    month_once_save_name = time.strftime('%Y-%m.pkl', time.localtime())
    save_path = os.path.join(config.model_save_path[prop], month_once_save_name)
    file_util.save_model(stacking_model, save_path)


def predict(prop, x):
    '''
        预测属性
    :param prop: config.model_save_path.keys()
    :param x:
    :return:
    '''
    if prop not in config.model_save_path.keys():
        return -1
    newest_model_path = file_util.get_newest_file(config.model_save_path[prop])
    model = file_util.load_model(newest_model_path)
    # x,_ = data_standardization(x)
    pred = model.predict(x)
    return pred


if __name__ == '__main__':
    for prop in config.data_load_path.keys():
        train(prop, k_fold=1, test_size=0.2)
    # x = np.array([[0, 0, 0, 20, 0, 0, 0.8, 48.8, 0, 20, 10.4, 0]])
    # print(predict("密度", x))
