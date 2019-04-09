#!/bin/bash
current=`date "+%Y%m%d_%H%M%S"`
python tc_run.py 'testB/testB_record_2019-01-26.csv' 'testB/testB_submit_2019-01-27.csv' '/testB/7_testB_results_'$current'.csv' 'all_data_include_B.csv' '../submit/testB/model/lgb_'

