import os
import sys
import logging
import argparse
import json
from datetime import datetime, timedelta
import time
from pytz import timezone

import settings
import utils
import data_manager


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock_code', nargs='+')
    parser.add_argument('--ver', choices=['v1', 'v2'], default='v2')
    parser.add_argument('--rl_method', choices=['dqn', 'pg', 'ac', 'a2c', 'a3c'])
    parser.add_argument('--net', choices=['dnn', 'lstm', 'cnn'], default='dnn')
    parser.add_argument('--n_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--discount_factor', type=float, default=0.9)
    parser.add_argument('--start_epsilon', type=float, default=0.5)
    parser.add_argument('--balance', type=int, default=10000000)
    parser.add_argument('--num_epoches', type=int, default=100)
    parser.add_argument('--delayed_reward_threshold', type=float, default=0.05)
    parser.add_argument('--mini_batch_size', type=int, default=20)
    parser.add_argument('--backend', choices=['tensorflow', 'plaidml'], default='tensorflow')
    args = parser.parse_args()

    # Keras Backend 설정
    if args.backend == 'tensorflow':
        os.environ['KERAS_BACKEND'] = 'tensorflow'
    elif args.backend == 'plaidml':
        os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

    # 출력 경로 설정
    time_str = utils.get_time_str()
    output_path = os.path.join(
        settings.BASE_DIR, 
        'output/{}_{}_{}'.format(time_str, args.rl_method, args.net)
    )
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 파라미터 기록
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        f.write(json.dumps(vars(args)))
    
    # 로그 기록 설정
    file_handler = logging.FileHandler(filename=os.path.join(
        output_path, "{}.log".format(time_str)), encoding='utf-8')
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s",
                        handlers=[file_handler, stream_handler], level=logging.DEBUG)
        
    # 로그, Keras Backend 설정을 먼저하고 RLTrader 모듈들을 이후에 임포트해야 함
    from agent import Agent
    from learners import DQNLearner, PolicyGradientLearner, ActorCriticLearner, A2CLearner, A3CLearner

    # 모델 경로 준비
    policy_network_path = os.path.join(output_path, '{}_{}_policy_{}.h5'.format(args.rl_method, args.net, time_str))
    value_network_path = os.path.join(output_path, '{}_{}_value_{}.h5'.format(args.rl_method, args.net, time_str))

    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_unit = []
    list_max_trading_unit = []

    for stock_code in args.stock_code:
        # 차트 데이터, 학습 데이터 준비
        chart_data, training_data = data_manager.load_data(
            os.path.join(settings.BASE_DIR, 'data/{}/{}.csv'.format(args.ver, stock_code)), ver=args.ver)
        
        # 최소/최대 투자 단위 설정
        min_trading_unit = max(int(100000 / chart_data.iloc[-1]['close']), 1)
        max_trading_unit = max(int(1000000 / chart_data.iloc[-1]['close']), 1)

        # 강화학습 시작
        learner = None
        if args.rl_method != 'a3c':
            # 공통 파라미터 설정
            common_params = {'rl_method': args.rl_method, 'stock_code': stock_code,
                            'chart_data': chart_data, 'training_data': training_data,
                            'min_trading_unit': min_trading_unit, 'max_trading_unit': max_trading_unit,
                            'delayed_reward_threshold': args.delayed_reward_threshold,
                            'mini_batch_size': args.mini_batch_size,
                            'net': args.net, 'n_steps': args.n_steps, 'lr': args.lr,
                            'output_path': output_path}
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
            if learner is not None:
                learner.run(balance=args.balance, num_epoches=args.num_epoches, 
                            discount_factor=args.discount_factor, start_epsilon=args.start_epsilon)
                learner.save_models()
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

    if args.rl_method == 'a3c' and learner is not None:
        learner.run(balance=args.balance, num_epoches=args.num_epoches, 
                    discount_factor=args.discount_factor, start_epsilon=args.start_epsilon)
        learner.save_models()
    
    sys.exit(0)