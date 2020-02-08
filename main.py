import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
import time
from pytz import timezone

import settings
import data_manager
import data_manager_v2

# 로그 기록 설정
log_dir = os.path.join(settings.BASE_DIR, 'logs')
time_str = settings.get_time_str()
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
file_handler = logging.FileHandler(filename=os.path.join(
    log_dir, "{}.log".format(time_str)), encoding='utf-8')
stream_handler = logging.StreamHandler(sys.stdout)
file_handler.setLevel(logging.DEBUG)
stream_handler.setLevel(logging.INFO)
logging.basicConfig(format="%(message)s",
                    handlers=[file_handler, stream_handler], level=logging.DEBUG)

# 로그 설정을 먼저하고 로깅하는 모듈들을 이후에 임포트해야 함
from agent import Agent
from learners import DQNLearner, PolicyGradientLearner, ActorCriticLearner, A2CLearner, A3CLearner


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock_codes', nargs='+')
    parser.add_argument('--ver', choices=['v1', 'v2'], default='v2')
    parser.add_argument('--rl_method', choices=['dqn', 'pg', 'ac', 'a2c', 'a3c'])
    parser.add_argument('--net', choices=['dnn', 'lstm', 'cnn'], default='dnn')
    parser.add_argument('--n_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--discount_factor', type=float, default=0.8)
    parser.add_argument('--start_epsilon', type=float, default=0.3)
    parser.add_argument('--num_epoches', type=int, default=100)
    parser.add_argument('--delayed_reward_threshold', type=float, default=0.05)
    parser.add_argument('--mini_batch_size', type=int, default=20)
    args = parser.parse_args()
    
    # 모델 경로 준비
    model_dir = os.path.join(settings.BASE_DIR, 'models')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    curr = datetime.fromtimestamp(time.time(), timezone('Asia/Seoul')).strftime("%Y%m%d%H%M%S")
    policy_network_path = os.path.join(model_dir, '{}_{}_p_{}.h5'.format(args.rl_method, args.net, curr))
    value_network_path = os.path.join(model_dir, '{}_{}_v_{}.h5'.format(args.rl_method, args.net, curr))

    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_unit = []
    list_max_trading_unit = []

    for stock_code in args.stock_codes:
        # 주식 데이터 준비
        chart_data, training_data = [], []
        if args.ver == 'v1':
            chart_data = data_manager.load_chart_data(
                os.path.join(settings.BASE_DIR,
                            'data/v1/chart_data/{}.csv'.format(stock_code)))
            prep_data = data_manager.preprocess(chart_data)
            training_data = data_manager.build_training_data(prep_data)

            # 기간 필터링
            training_data = training_data[(training_data['date'] >= '2017-01-01') &
                                        (training_data['date'] <= '2017-12-31')]
            training_data = training_data.dropna()
                
            # 차트 데이터 분리
            features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']
            chart_data = training_data[features_chart_data]

            # 학습 데이터 분리
            features_training_data = [
                'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
                'close_lastclose_ratio', 'volume_lastvolume_ratio',
                'close_ma5_ratio', 'volume_ma5_ratio',
                'close_ma10_ratio', 'volume_ma10_ratio',
                'close_ma20_ratio', 'volume_ma20_ratio',
                'close_ma60_ratio', 'volume_ma60_ratio',
                'close_ma120_ratio', 'volume_ma120_ratio',
            ]
            training_data = training_data[features_training_data]
        elif args.ver == 'v2':
            chart_data, training_data = data_manager_v2.load_data(
                os.path.join(settings.BASE_DIR, 'data/v2/{}.csv'.format(stock_code))
            )

            # 기간 필터링
            chart_data = chart_data[(chart_data['date'] >= '20170101') &
                                        (chart_data['date'] <= '20171231')]
            training_data = training_data[(training_data['date'] >= '20170101') &
                                        (training_data['date'] <= '20171231')]
            training_data = training_data.dropna()
            training_data = training_data.drop(columns=['date'])

        # 최소/최대 투자 단위 설정
        min_trading_unit = max(int(100000 / chart_data.iloc[-1]['close']), 1)
        # max_trading_unit = max(int(1000000 / chart_data.iloc[-1]['close']), 1)
        max_trading_unit = max(int(100000 / chart_data.iloc[-1]['close']), 1)

        # 강화학습 시작
        learner = None
        if args.rl_method != 'a3c':
            # 공통 파라미터 설정
            common_params = {'rl_method': args.rl_method, 'stock_code': stock_code,
                            'chart_data': chart_data, 'training_data': training_data,
                            'min_trading_unit': min_trading_unit, 'max_trading_unit': max_trading_unit,
                            'delayed_reward_threshold': args.delayed_reward_threshold,
                            'mini_batch_size': args.mini_batch_size,
                            'net': args.net, 'n_steps': args.n_steps, 'lr': args.lr}
            if args.rl_method == 'dqn':
                learner = DQNLearner(**{**common_params, 'value_network_path': value_network_path})
            elif args.rl_method == 'pg':
                learner = PolicyGradientLearner(**{**common_params, 'policy_network_path': policy_network_path})
            elif args.rl_method == 'ac':
                learner = ActorCriticLearner(**{
                    **common_params, 
                    'value_network_path': value_network_path, 'policy_network_path': policy_network_path})
            elif args.rl_method == 'a2c':
                learner = A2CLearner(**{
                    **common_params, 
                    'value_network_path': value_network_path, 'policy_network_path': policy_network_path})
        else:
            list_stock_code.append(stock_code)
            list_chart_data.append(chart_data)
            list_training_data.append(training_data)
            list_min_trading_unit.append(min_trading_unit)
            list_max_trading_unit.append(max_trading_unit)
            learner = A3CLearner(
                list_stock_code=list_stock_code, list_chart_data=list_chart_data, list_training_data=list_training_data,
                list_min_trading_unit=list_min_trading_unit, list_max_trading_unit=list_max_trading_unit,
                delayed_reward_threshold=args.delayed_reward_threshold, 
                net=args.net, n_steps=args.n_steps, lr=args.lr,
                value_network_path=value_network_path, policy_network_path=policy_network_path)

        if learner is not None:
            learner.fit(balance=10000000, num_epoches=args.num_epoches, 
                        discount_factor=args.discount_factor, start_epsilon=args.start_epsilon)
            # 신경망을 파일로 저장
            learner.save_models()