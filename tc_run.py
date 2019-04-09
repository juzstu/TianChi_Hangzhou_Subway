#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:juzphy
# datetime:2019-04-01 10:26

'''
python version：3.7.2
platform：Mac OS Mojave 10.14.3
lightgbm version: 2.2.3
'''

import pandas as pd
from datetime import timedelta
import os
import numpy as np
import lightgbm as lgb
import warnings
from collections import defaultdict
import gensim
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings('ignore')

data_path = '../data/'
sub_path = '../submit/'
if not os.path.exists(sub_path + 'testB'):
    os.mkdir(sub_path + 'testB')
if not os.path.exists(sub_path + 'testC'):
    os.mkdir(sub_path + 'testC')
if not os.path.exists(sub_path + 'testB/model'):
    os.mkdir(sub_path + 'testB/model')
if not os.path.exists(sub_path + 'testC/model'):
    os.mkdir(sub_path + 'testC/model')
cnt_dict = {0: 'outNums', 1: 'inNums'}


def gen_data(provide_test_path, save_path):
    all_data_in = pd.DataFrame()
    all_data_out = pd.DataFrame()
    file_list = [data_path + '/train/record_2019-01-' + str(c).zfill(2) + '.csv' for c in range(1, 26)]
    file_list.append(data_path + provide_test_path)
    for f in file_list:
        print(f'Start merge file of {f} ...')
        train = pd.read_csv(f)
        train['time'] = pd.to_datetime(train['time'])
        train['minutes'] = train['time'].dt.minute
        train = train.set_index('time')
        for j in range(2):
            for i in range(81):
                if i != 54:
                    tmp = train[(train['stationID'] == i) & (train['status'] == j)]
                    tmp_df = pd.DataFrame(tmp.resample('10T', closed='left')['stationID'].count().reset_index())

                    tmp_df.columns = ['startTime', cnt_dict[j]]
                    tmp_df[f'line_unique_for_{cnt_dict[j]}'] = tmp.resample('10T', closed='left')['lineID'].nunique().values
                    tmp_df[f'device_unique_for_{cnt_dict[j]}'] = tmp.resample('10T', closed='left')['deviceID'].nunique().values
                    tmp_df[f'pay_unique_for_{cnt_dict[j]}'] = tmp.resample('10T', closed='left')['payType'].nunique().values
                    tmp_df[f'minute_unique_for_{cnt_dict[j]}'] = tmp.resample('10T', closed='left')['minutes'].nunique().values
                    tmp_df[f'minute_min_for_{cnt_dict[j]}'] = tmp.resample('10T', closed='left')['minutes'].min().values
                    tmp_df[f'minute_max_for_{cnt_dict[j]}'] = tmp.resample('10T', closed='left')['minutes'].max().values
                    tmp_df[f'minute_mean_for_{cnt_dict[j]}'] = tmp.resample('10T', closed='left')['minutes'].mean().values
                    tmp_df[f'minute_skew_for_{cnt_dict[j]}'] = tmp.resample('10T', closed='left')['minutes'].skew().values
                    first_value = tmp_df.iloc[0]['startTime']
                    first_time = pd.to_datetime(str(first_value).split(' ')[0] + ' 00:00:00')
                    last_value = tmp_df.iloc[-1]['startTime']
                    last_time = pd.to_datetime(str(first_value).split(' ')[0] + ' 23:50:00')
                    if first_value > first_time:
                        add_tmp_first = pd.DataFrame()
                        add_tmp_first['startTime'] = pd.date_range(start=first_time,
                                                                   end=first_value - timedelta(minutes=10), freq='10T')
                        add_tmp_first[cnt_dict[j]] = 0
                        add_tmp_first[f'line_unique_for_{cnt_dict[j]}'] = 0
                        add_tmp_first[f'device_unique_for_{cnt_dict[j]}'] = 0
                        add_tmp_first[f'pay_unique_for_{cnt_dict[j]}'] = 0
                        add_tmp_first[f'minute_unique_for_{cnt_dict[j]}'] = 0
                        add_tmp_first[f'minute_min_for_{cnt_dict[j]}'] = 0
                        add_tmp_first[f'minute_max_for_{cnt_dict[j]}'] = 0
                        add_tmp_first[f'minute_mean_for_{cnt_dict[j]}'] = 0
                        add_tmp_first[f'minute_skew_for_{cnt_dict[j]}'] = 0
                        tmp_df = add_tmp_first.append(tmp_df)
                    if last_value < last_time:
                        add_tmp_last = pd.DataFrame()
                        add_tmp_last['startTime'] = pd.date_range(start=last_value + timedelta(minutes=10),
                                                                  end=last_time, freq='10T')
                        add_tmp_last[cnt_dict[j]] = 0
                        add_tmp_last[f'line_unique_for_{cnt_dict[j]}'] = 0
                        add_tmp_last[f'device_unique_for_{cnt_dict[j]}'] = 0
                        add_tmp_last[f'pay_unique_for_{cnt_dict[j]}'] = 0
                        add_tmp_last[f'minute_unique_for_{cnt_dict[j]}'] = 0
                        add_tmp_last[f'minute_min_for_{cnt_dict[j]}'] = 0
                        add_tmp_last[f'minute_max_for_{cnt_dict[j]}'] = 0
                        add_tmp_last[f'minute_mean_for_{cnt_dict[j]}'] = 0
                        add_tmp_last[f'minute_skew_for_{cnt_dict[j]}'] = 0
                        tmp_df = tmp_df.append(add_tmp_last)
                    tmp_df['endTime'] = tmp_df['startTime'] + timedelta(minutes=10)
                    tmp_df['stationID'] = i
                    if tmp_df.shape[0] == 144:
                        if j == 0:
                            all_data_in = all_data_in.append(tmp_df)
                        else:
                            all_data_out = all_data_out.append(tmp_df)
                    else:
                        print('data shape not correct')

    all_data = pd.merge(all_data_in, all_data_out, on=['stationID', 'startTime', 'endTime'])
    all_data['dayofweek'] = all_data['startTime'].dt.dayofweek.values + 1
    all_data['day'] = all_data['startTime'].dt.day.values
    all_data['hour'] = all_data['startTime'].dt.hour.values
    all_data.fillna(0, inplace=True)
    print(f'Final data shape: {all_data.shape}')
    all_data.sort_values(['stationID', 'startTime'], inplace=True)
    all_data.to_csv(data_path + save_path, encoding='utf8', index=False)


def get_n2v_feats(path):
    model = gensim.models.Word2Vec.load(path)
    n2v_list = []
    for i in range(0, 81):
        n2v_list.append(model[str(i)])

    nodes = pd.DataFrame(np.vstack(n2v_list))
    nodes.columns = nodes.columns.map(lambda x: 'stationID_n2v_' + str(x))
    nodes.reset_index(inplace=True)
    nodes.rename(columns={'index': 'stationID'}, inplace=True)
    return nodes


def gen_feats(data_frame, use_feats, station_dict):
    day_unique = data_frame['day'].nunique()
    print(data_frame['day'].unique())
    use_train_df = pd.DataFrame()
    data_frame['pre_num'] = data_frame['day'].map(dict(zip(data_frame['day'].unique(), range(day_unique))))
    print('gen feats ...')
    for i in range(1, day_unique):
        print(f'start add index of {i}')
        pre_df = data_frame[data_frame['pre_num'] <= i - 1][
            ['stationID', 'inNums', 'outNums', 'same_period', 'pre_num']]
        group_df_period = pre_df.groupby(['stationID', 'same_period'])[['inNums', 'outNums']].agg(
            ['min', 'max', 'mean', 'skew']).reset_index()
        group_df_period.columns = ['stationID', 'same_period'] + [f'period_{p}_{q}' for p in ['for_in', 'for_out'] for q
                                                                  in ['min', 'max', 'mean', 'skew']]

        tmp = data_frame[data_frame['pre_num'] == i - 1][use_feats]
        tmp['shift_b_for_in'] = data_frame[data_frame['pre_num'] == i - 1]['inNums'].shift(1).values
        tmp['shift_f_for_in'] = data_frame[data_frame['pre_num'] == i - 1]['inNums'].shift(-1).values
        tmp['shift_b_for_out'] = data_frame[data_frame['pre_num'] == i - 1]['outNums'].shift(1).values
        tmp['shift_f_for_out'] = data_frame[data_frame['pre_num'] == i - 1]['outNums'].shift(-1).values

        # 前一天的60min, 120min的历史滑窗
        for t in [6, 12]:
            tmp[f'roll_{t}_for_in_mean'] = data_frame[data_frame['pre_num'] == i - 1]['inNums'].rolling(
                window=t).mean().values
            tmp[f'roll_{t}_for_out_mean'] = data_frame[data_frame['pre_num'] == i - 1]['outNums'].rolling(
                window=t).mean().values
            tmp[f'ewm_{t}_for_in_mean'] = data_frame[data_frame['pre_num'] == i - 1]['inNums'].ewm(span=t).mean().values
            tmp[f'ewm_{t}_for_out_mean'] = data_frame[data_frame['pre_num'] == i - 1]['outNums'].ewm(
                span=t).mean().values

            tmp[f'roll_{t}_for_in_mean_center'] = data_frame[data_frame['pre_num'] == i - 1]['inNums'].\
                rolling(window=t, center=True).mean().values
            tmp[f'roll_{t}_for_out_mean_center'] = data_frame[data_frame['pre_num'] == i - 1]['outNums'].rolling(
                window=t, center=True).mean().values
        # 连接的其他station的历史统计
        con_df = pd.DataFrame()
        for c in data_frame['stationID'].unique():
            con_tmp = data_frame[(data_frame['pre_num'] <= i - 1) & (data_frame['stationID'].isin(station_dict[c]))][
                ['inNums', 'outNums', 'same_period']]
            con_tmp = con_tmp.groupby('same_period')[['inNums', 'outNums']].agg(
                ['min', 'max', 'mean', 'std', 'skew']).reset_index()
            con_tmp.columns = ['same_period'] + [f'con_{p}_{q}' for p in ['for_in', 'for_out'] for q in
                                                 ['min', 'max', 'mean', 'std', 'skew']]
            con_tmp['stationID'] = c
            con_df = con_df.append(con_tmp)
        tmp['startTime'] = data_frame[data_frame['pre_num'] == i]['startTime'].values
        tmp['inNums'] = data_frame[data_frame['pre_num'] == i]['inNums'].values
        tmp['outNums'] = data_frame[data_frame['pre_num'] == i]['outNums'].values

        tmp = tmp.merge(group_df_period, on=['stationID', 'same_period'], how='left')
        tmp = tmp.merge(con_df, on=['stationID', 'same_period'], how='left')
        del tmp['same_period']
        use_train_df = use_train_df.append(tmp)
    use_train_df['day'] = use_train_df['startTime'].dt.day.values
    use_train_df['dayofweek'] = use_train_df['startTime'].dt.dayofweek.values + 1
    use_train_df['hour'] = use_train_df['startTime'].dt.hour.values
    use_train_df['minute'] = use_train_df['startTime'].dt.minute.values
    return use_train_df


def lgb_train(train_, valid_, pred, label, fraction):
    print(f'data shape:\ntrain--{train_.shape}\nvalid--{valid_.shape}')
    print(f'days:\ntrain--{train_.day.sort_values().unique()}\nvalid--{valid_.day.unique()}')
    print(train_['dayofweek'].unique())
    print(f'Use {len(pred)} features ...')
    params = {
        'learning_rate': 0.03,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 63,
        'feature_fraction': fraction,
        'bagging_fraction': fraction,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
        'reg_sqrt': True,
        'nthread': -1,
        'verbose': -1,
    }

    dtrain = lgb.Dataset(train_[pred], label=train_[label], categorical_feature=['stationID'])
    dvalid = lgb.Dataset(valid_[pred], label=valid_[label], categorical_feature=['stationID'])

    clf = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=10000,
        valid_sets=[dvalid],
        early_stopping_rounds=100,
        verbose_eval=500
    )
    return clf


def lgb_sub_all(train_, test_, pred, label, num_rounds, fraction, add_rounds, model_file):
    print(f'data shape:\ntrain--{train_.shape}\ntest--{test_.shape}')
    print(f'days:\ntrain--{train_.day.sort_values().unique()}\ntest--{test_.day.unique()}')
    print(f'Use {len(pred)} features and {num_rounds} rounds...')
    params = {
        'learning_rate': 0.03,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 63,
        'feature_fraction': fraction,
        'bagging_fraction': fraction,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
        'reg_sqrt': True,
        'nthread': -1,
        'verbose': -1,
    }

    dtrain = lgb.Dataset(train_[pred], label=train_[label], categorical_feature=['stationID'])

    clf = lgb.train(
        params=params,
        train_set=dtrain,
        valid_sets=[dtrain],
        num_boost_round=num_rounds+add_rounds,
        verbose_eval=500
    )
    test_[label] = clf.predict(test_[pred])
    clf.save_model(f'{model_file}_{label}.txt')


if __name__ == "__main__":
    test_pre_day = sys.argv[1]
    test_df_path = sys.argv[2]
    sub_file = sys.argv[3]
    df_save_path = sys.argv[4]
    model_path = sys.argv[5]
    node_path = 'station_matrix.m'
    if not os.path.exists(data_path + df_save_path):
        gen_data(test_pre_day, df_save_path)

    # get station dict
    map_data = pd.read_csv(data_path + 'roadMap.csv', names=[f'station_{i}' for i in range(81)], header=0)
    tmp_dict = defaultdict(list)
    for i in range(map_data.shape[0]):
        for j in map_data.columns:
            if map_data.loc[i, j] == 1:
                tmp_dict[i].append(int(j.split('_')[-1]))

    all_df = pd.read_csv(data_path + df_save_path)
    all_df['startTime'] = pd.to_datetime(all_df['startTime'])
    all_df['same_period'] = all_df['startTime'].apply(lambda x: str(x).split(' ')[-1])

    test_origin = pd.read_csv(data_path + test_df_path)
    test_origin['startTime'] = pd.to_datetime(test_origin['startTime'])
    test = test_origin.copy()
    test = test[test['stationID'] != 54]
    test['same_period'] = test['startTime'].apply(lambda x: str(x).split(' ')[-1])
    test['dayofweek'] = test['startTime'].dt.dayofweek.values + 1
    test['day'] = test['startTime'].dt.day.values

    test_day = test['startTime'].dt.day.unique()[0]
    print('test day:', test_day)

    test_weekday = test['startTime'].dt.dayofweek.unique()[0] + 1

    all_df = all_df.append(test)
    node_df = get_n2v_feats(node_path)
    all_df = pd.merge(all_df, node_df, on=['stationID'], how='left')

    all_df = all_df[all_df['day'] > 1]

    max_weekday = all_df['dayofweek'].unique().max()

    df_feats = [c for c in all_df.columns if
                c not in ['pre_num', 'inNums', 'outNums', 'startTime', 'endTime', 'dayofweek', 'day', 'hour', 'minute']]
    node_list = [f'stationID_n2v_{s}' for s in range(8)]

    # way 1
    use_df_way1 = pd.DataFrame()
    for i in range(1, max_weekday+1):
        df_tmp = gen_feats(all_df[all_df['dayofweek'] == i], df_feats, tmp_dict)
        use_df_way1 = use_df_way1.append(df_tmp)
    use_df_way1 = use_df_way1.sample(frac=1, random_state=1)

    use_train_way1 = use_df_way1[use_df_way1['day'] != test_day]
    use_test_way1 = use_df_way1[use_df_way1['day'] == test_day]

    in_feats_way1 = ['line_unique_for_inNums', 'device_unique_for_inNums', 'pay_unique_for_inNums', 'roll_6_for_in_mean',
                     'period_for_in_mean', 'con_for_in_min', 'con_for_in_max', 'con_for_in_mean', 'con_for_in_std',
                     'stationID', 'dayofweek', 'day', 'hour', 'minute'] + node_list

    out_feats_way1 = [t for t in use_test_way1.columns if 'for_out' in t and t not in ['con_for_out_std']] + node_list\
                     + ['dayofweek', 'day', 'hour', 'stationID', 'minute']

    use_valid_way1 = use_train_way1[use_train_way1['day'] == (test_day - 1)]
    for_train_way1 = use_train_way1[use_train_way1['day'] != (test_day - 1)]

    if test_weekday > 5:
        for_in_train_way1 = for_train_way1[(for_train_way1['dayofweek'].isin([2, 3, 4, 5, 6, 7]))]
        for_out_train_way1 = for_train_way1[(for_train_way1['dayofweek'].isin([3, 4, 5, 6, 7]))]
    else:
        for_in_train_way1 = for_train_way1.copy()
        for_out_train_way1 = for_train_way1.copy()

    in_model_way1 = lgb_train(for_in_train_way1, use_valid_way1, in_feats_way1, 'inNums', 0.8)
    lgb_sub_all(for_in_train_way1.append(use_valid_way1), use_test_way1, in_feats_way1, 'inNums',
                in_model_way1.best_iteration, 0.8, 30, model_path+'way1')
    out_model_way1 = lgb_train(for_out_train_way1, use_valid_way1, out_feats_way1, 'outNums', 0.8)
    lgb_sub_all(for_out_train_way1.append(use_valid_way1), use_test_way1, out_feats_way1, 'outNums',
                out_model_way1.best_iteration, 0.8, 30, model_path+'way1')

    submit_way1 = pd.merge(test_origin[['stationID', 'startTime', 'endTime']],
                           use_test_way1[['stationID', 'startTime', 'inNums', 'outNums']], on=['stationID', 'startTime'],
                           how='left')

    submit_way1.fillna(0, inplace=True)
    submit_way1.loc[submit_way1['inNums'] < 0, 'inNums'] = 0
    submit_way1.loc[submit_way1['outNums'] < 0, 'outNums'] = 0
    submit_way1[['stationID', 'inNums', 'outNums']] = submit_way1[['stationID', 'inNums', 'outNums']].round().astype(int)

    # way 2
    if test_weekday > 5:
        use_df_way2 = gen_feats(all_df, df_feats, tmp_dict)
    else:
        print('test day is not weekend ...')
        use_df_way2 = gen_feats(all_df[~all_df['dayofweek'].isin([6, 7])], df_feats, tmp_dict)
    use_train_way2 = use_df_way2[use_df_way2['day'] != test_day]
    use_test_way2 = use_df_way2[use_df_way2['day'] == test_day]
    in_feats_way2 = ['line_unique_for_inNums', 'device_unique_for_inNums', 'pay_unique_for_inNums', 'roll_6_for_in_mean',
                     'period_for_in_mean', 'con_for_in_min', 'con_for_in_max', 'con_for_in_mean', 'con_for_in_std',
                     'stationID', 'dayofweek', 'day', 'hour', 'minute']

    out_feats_way2 = ['line_unique_for_outNums', 'device_unique_for_outNums', 'pay_unique_for_outNums',
                      'minute_unique_for_outNums', 'minute_min_for_outNums', 'minute_max_for_outNums',
                      'minute_mean_for_outNums', 'minute_skew_for_outNums', 'shift_b_for_out', 'shift_f_for_out',
                      'roll_6_for_out_mean', 'ewm_6_for_out_mean', 'roll_6_for_out_mean_center', 'roll_12_for_out_mean',
                      'ewm_12_for_out_mean', 'roll_12_for_out_mean_center', 'period_for_out_min', 'period_for_out_max',
                      'period_for_out_mean', 'period_for_out_skew', 'con_for_out_min', 'con_for_out_max',
                      'con_for_out_mean', 'con_for_out_skew', 'dayofweek', 'day', 'hour', 'stationID', 'minute']

    use_valid_way2 = use_train_way2[use_train_way2['day'] == (test_day - 1)]
    for_train_way2 = use_train_way2[use_train_way2['day'] != (test_day - 1)]

    in_model_way2 = lgb_train(for_train_way2, use_valid_way2, in_feats_way2, 'inNums', 0.9)
    lgb_sub_all(use_train_way2, use_test_way2, in_feats_way2, 'inNums', in_model_way2.best_iteration, 0.9, 30, model_path+'way2')
    out_model_way2 = lgb_train(for_train_way2, use_valid_way2, out_feats_way2, 'outNums', 0.8)
    lgb_sub_all(use_train_way2, use_test_way2, out_feats_way2, 'outNums', out_model_way2.best_iteration, 0.8, 30, model_path+'way2')

    submit_way2 = pd.merge(test_origin[['stationID', 'startTime', 'endTime']],
                           use_test_way2[['stationID', 'startTime', 'inNums', 'outNums']], on=['stationID', 'startTime'],
                           how='left')

    submit_way2.fillna(0, inplace=True)
    submit_way2.loc[submit_way2['inNums'] < 0, 'inNums'] = 0
    submit_way2.loc[submit_way2['outNums'] < 0, 'outNums'] = 0
    submit_way2[['stationID', 'inNums', 'outNums']] = submit_way2[['stationID', 'inNums', 'outNums']].round().astype(int)
  
    # final result 6:4 ensemble
    if test_day > 29:
        print('start ensemble ...')
        submit_way1['inNums'] = submit_way1['inNums'] * 0.4 + submit_way2['inNums'] * 0.6
        submit_way1['outNums'] = submit_way1['outNums'] * 0.4 + submit_way2['outNums'] * 0.6
        submit_way1['inNums'] = submit_way1['inNums'] * 0.96
        submit_way1['outNums'] = submit_way1['outNums'] * 0.96
    else:
        submit_way1['inNums'] = submit_way1['inNums'] * 0.6 + submit_way2['inNums'] * 0.4
        submit_way1['outNums'] = submit_way1['outNums'] * 0.6 + submit_way2['outNums'] * 0.4
    submit_way1.to_csv(sub_path + sub_file, encoding='utf8', index=False)
