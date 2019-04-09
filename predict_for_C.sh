#!/bin/bash
current=`date "+%Y%m%d_%H%M%S"`
python tc_run.py 'testC/testC_record_2019-01-30.csv' 'testC/testC_submit_2019-01-31.csv' '/testC/7_testC_results_'$current'.csv' 'all_data_include_C.csv' '../submit/testC/model/lgb_'

