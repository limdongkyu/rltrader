#-*- coding:utf-8 -*-
import os
import locale
import logging
import abc
import collections
import threading
import time
import numpy as np
from utils import sigmoid
from environment import Environment
from agent import Agent
from networks import Network, DNN, LSTMNetwork, CNN
from visualizer import Visualizer


class ReinforcementLearner:
    __metaclass__ = abc.ABCMeta

    def __init__(self, rl_method='rl', stock_code=None, chart_data=None, training_data=None,
                 min_trading_unit=1, max_trading_unit=2, 
                 delayed_reward_threshold=.05, mini_batch_size=0,
                 net='dnn', n_steps=1, lr=0.01,
                 value_network=None, policy_network=None,
                 output_path=''):
        self.rl_method = rl_method
        self.stock_code = stock_code
        # 환경 설정
        self.chart_data = chart_data
        self.environment = Environment(chart_data)
        # 에이전트 설정
        self.agent = Agent(self.environment,
                           min_trading_unit=min_trading_unit,
                           max_trading_unit=max_trading_unit,
                           delayed_reward_threshold=delayed_reward_threshold)
        # 학습 데이터
        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1
        self.mini_batch_size = mini_batch_size
        # 총 자질 벡터 크기 = 학습 데이터의 자질 벡터 크기 + 에이전트 상태 크기
        self.num_features = self.agent.STATE_DIM
        if self.training_data is not None:
            self.num_features += self.training_data.shape[1]
        # 신경망 설정
        self.net = net
        self.n_steps = n_steps
        self.lr = lr
        self.value_network = value_network
        self.policy_network = policy_network
        # 가시화 모듈
        self.visualizer = Visualizer()
        # 메모리
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []
        # 에포크 관련 정보
        self.loss = 0.
        self.itr_cnt = 0
        self.win_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        self.pos_learning_cnt = 0
        self.neg_learning_cnt = 0
        # 로그 등 출력 경로
        self.output_path = output_path

    def init_value_network(self, shared_network=None, activation='tanh'):
        if self.net == 'dnn':
            self.value_network = DNN(
                input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, lr=self.lr, shared_network=shared_network, activation=activation)
        elif self.net == 'lstm':
            self.value_network = LSTMNetwork(
                input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, lr=self.lr, n_steps=self.n_steps, shared_network=shared_network, activation=activation)
        elif self.net == 'cnn':
            self.value_network = CNN(
                input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, lr=self.lr, n_steps=self.n_steps, shared_network=shared_network, activation=activation)
        if self.value_network_path is not None and os.path.exists(self.value_network_path):
            self.value_network.load_model(model_path=self.value_network_path)

    def init_policy_network(self, shared_network=None, activation='sigmoid'):
        if self.net == 'dnn':
            self.policy_network = DNN(
                input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, lr=self.lr, shared_network=shared_network, activation=activation)
        elif self.net == 'lstm':
            self.policy_network = LSTMNetwork(
                input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, lr=self.lr, n_steps=self.n_steps, shared_network=shared_network, activation=activation)
        elif self.net == 'cnn':
            self.policy_network = CNN(
                input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, lr=self.lr, n_steps=self.n_steps, shared_network=shared_network, activation=activation)
        if self.policy_network_path is not None and os.path.exists(self.policy_network_path):
            self.policy_network.load_model(model_path=self.policy_network_path)

    def reset(self):
        self.sample = None
        self.training_data_idx = -1
        self.itr_cnt = 0
        self.exploration_cnt = 0
        # 환경 초기화
        self.environment.reset()
        # 에이전트 초기화
        self.agent.reset()
        # 가시화 초기화
        self.visualizer.clear([0, len(self.chart_data)])
        # 메모리 초기화
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []

    def visualize(self, epoch_str, num_epoches, epsilon):
        self.visualizer.plot(
            epoch_str=epoch_str, num_epoches=num_epoches, epsilon=epsilon,
            action_list=Agent.ACTIONS, actions=list(self.memory_action),
            num_stocks=list(self.memory_num_stocks), 
            outvals_value=list(self.memory_value), outvals_policy=list(self.memory_policy),
            exps=list(self.memory_exp_idx), learning=list(self.memory_learning_idx),
            initial_balance=self.agent.initial_balance, pvs=list(self.memory_pv)
        )
        self.visualizer.save(os.path.join(
            self.epoch_summary_dir, 'epoch_summary_{}.png'.format(epoch_str)))

    def fit(self, delayed_reward, discount_factor, full=False):
        # 배치 학습 데이터 크기
        if full:
            self.batch_size = len(self.memory_sample)
        elif self.mini_batch_size > 0:
            self.batch_size = min(self.batch_size, self.mini_batch_size)
        # 배치 학습 데이터 생성 및 신경망 갱신
        if self.batch_size > 0:
            _loss = self.update_networks(self.batch_size, delayed_reward, discount_factor)
            if _loss is not None:
                self.loss += abs(_loss)
                if delayed_reward > 0:
                    self.pos_learning_cnt += 1
                else:
                    self.neg_learning_cnt += 1
                self.memory_learning_idx.append([self.training_data_idx, delayed_reward])
            self.batch_size = 0

    def run(
        self, num_epoches=100, balance=10000000,
        discount_factor=0.9, start_epsilon=1, learning=True):
        info = "[{code}] RL:{rl} Net:{net} LR:{lr} DF:{discount_factor} TU:[{min_trading_unit},{max_trading_unit}] DRT:{delayed_reward_threshold}".format(
            code=self.stock_code, rl=self.rl_method, net=self.net,
            lr=self.lr, discount_factor=discount_factor,
            min_trading_unit=self.agent.min_trading_unit, max_trading_unit=self.agent.max_trading_unit,
            delayed_reward_threshold=self.agent.delayed_reward_threshold
        )
        logging.info(info)

        # 시작 시간
        time_start = time.time()

        # 가시화 준비
        # 차트 데이터는 변하지 않으므로 미리 가시화
        self.visualizer.prepare(self.environment.chart_data, info)

        # 가시화 결과 저장할 폴더 준비
        self.epoch_summary_dir = os.path.join(self.output_path, 'epoch_summary')
        if not os.path.isdir(self.epoch_summary_dir):
            os.makedirs(self.epoch_summary_dir)

        # 에이전트 초기 자본금 설정
        self.agent.set_balance(balance)

        # 학습에 대한 정보 초기화
        max_portfolio_value = 0
        epoch_win_cnt = 0

        # 학습 반복
        for epoch in range(num_epoches):
            # step 샘플을 만들기 위한 큐
            q_sample = collections.deque(maxlen=self.n_steps)
            
            # 환경, 에이전트, 신경망, 가시화, 메모리 초기화
            self.reset()

            # 학습을 진행할 수록 탐험 비율 감소
            if learning:
                epsilon = start_epsilon * (1. - float(epoch) / (num_epoches - 1))
            else:
                epsilon = 0

            while True:
                # 샘플 생성
                next_sample = self.build_sample()
                if next_sample is None:
                    break

                # n_step만큼 샘플 저장
                q_sample.append(next_sample)
                if len(q_sample) < self.n_steps:
                    continue

                # 가치, 정책 신경망 예측
                pred_value = None
                pred_policy = None
                if self.value_network is not None:
                    pred_value = self.value_network.predict(list(q_sample))
                if self.policy_network is not None:
                    pred_policy = self.policy_network.predict(list(q_sample))
                
                # 신경망 또는 탐험에 의한 행동 결정
                pred = pred_policy if pred_policy is not None else pred_value
                action, confidence, exploration = self.agent.decide_action(pred, epsilon)

                # 결정한 행동을 수행하고 즉시 보상과 지연 보상 획득
                immediate_reward, delayed_reward = self.agent.act(action, confidence)

                # 행동 및 행동에 대한 결과를 기억
                self.memory_sample.append(list(q_sample))
                self.memory_action.append(action)
                self.memory_reward.append(immediate_reward)
                if self.value_network is not None:
                    self.memory_value.append(pred_value)
                if self.policy_network is not None:
                    self.memory_policy.append(pred_policy)
                self.memory_pv.append(self.agent.portfolio_value)
                self.memory_num_stocks.append(self.agent.num_stocks)
                if exploration:
                    self.memory_exp_idx.append(self.training_data_idx)

                # 반복에 대한 정보 갱신
                self.batch_size += 1
                self.itr_cnt += 1
                self.exploration_cnt += 1 if exploration else 0
                self.win_cnt += 1 if delayed_reward > 0 else 0

                # 학습 모드이고 지연 보상이 존재할 경우 정책 신경망 갱신
                if self.mini_batch_size > 0 and delayed_reward == 0 and self.batch_size >= self.mini_batch_size:
                    delayed_reward = immediate_reward
                    self.agent.base_portfolio_value = self.agent.portfolio_value
                if learning and delayed_reward != 0:
                    self.fit(delayed_reward, discount_factor)

            # 에포크 관련 정보 가시화
            num_epoches_digit = len(str(num_epoches))
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')
            if epoch == 0 or (epoch + 1) % 10 == 0:
                self.visualize(epoch_str, num_epoches, epsilon)

            # 에포크 관련 정보 로그 기록
            if self.pos_learning_cnt + self.neg_learning_cnt > 0:
                self.loss /= self.pos_learning_cnt + self.neg_learning_cnt
            logging.info("[{}][Epoch {}/{}] Epsilon:%.4f #Expl.:{}/{} "
                        "#Buy:{} #Sell:{} #Hold:{} "
                        "#Stocks:{} PV:{:,} "
                        "POS:{} NEG:{} Loss:{:.6f}".format(
                            self.stock_code, epoch_str, num_epoches, epsilon, self.exploration_cnt, self.itr_cnt,
                            self.agent.num_buy, self.agent.num_sell, self.agent.num_hold,
                            self.agent.num_stocks,
                            self.agent.portfolio_value,
                            self.pos_learning_cnt, self.neg_learning_cnt, self.loss))

            # 학습 관련 정보 갱신
            max_portfolio_value = max(
                max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        # 에포크 종료 후 학습
        if learning:
            full = True if self.value_network is not None else False
            self.fit(self.agent.profitloss, discount_factor, full=full)

        # 종료 시간
        time_end = time.time()
        elapsed_time = time_end - time_start

        # 학습 관련 정보 로그 기록
        logging.info("[{code}] Elapsed Time: {elapsed_time}, Max PV: {max_pv}, \t #Win: {cnt_win}".format(
            code=self.stock_code, elapsed_time=elapsed_time, max_pv=locale.currency(max_portfolio_value, grouping=True), cnt_win=epoch_win_cnt))

    def build_sample(self):
        self.environment.observe()
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[self.training_data_idx].tolist()
            self.sample.extend(self.agent.get_states())
            return self.sample
        return None

    def get_action_network(self):
        if self.policy_network is not None:
            return self.policy_network
        else:
            return self.value_network

    @abc.abstractmethod
    def get_batch(self, batch_size, delayed_reward, discount_factor):
        pass

    def update_networks(self, batch_size, delayed_reward, discount_factor):
        # 배치 학습 데이터 생성
        x, y_value, y_policy = self.get_batch(batch_size, delayed_reward, discount_factor)
        if len(x) > 0:
            loss = 0
            if y_value is not None:
                # 가치 신경망 갱신
                loss += self.value_network.train_on_batch(x, y_value)
            if y_policy is not None:
                # 정책 신경망 갱신
                loss += self.policy_network.train_on_batch(x, y_policy)
            return loss
        return None

    def trade(self, model_path=None, balance=2000000):
        if model_path is None:
            return
        self.policy_network.load_model(model_path=model_path)
        self.run(balance=balance, num_epoches=1, learning=False)

    def save_models(self):
        if self.value_network is not None and self.value_network_path is not None:
            self.value_network.save_model(self.value_network_path)
        if self.policy_network is not None and self.policy_network_path is not None:
            self.policy_network.save_model(self.policy_network_path)


class DQNLearner(ReinforcementLearner):
    def __init__(self, *args, value_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_network_path = value_network_path
        self.init_value_network()

    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(
            reversed(self.memory_sample[-batch_size:]),
            reversed(self.memory_action[-batch_size:]),
            reversed(self.memory_value[-batch_size:]),
            reversed(self.memory_reward[-(batch_size-1):] + [0]),
        )
        x = np.zeros((batch_size, self.n_steps, self.num_features))
        y = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        value_max_next = 0
        for i, (sample, action, value, reward) in enumerate(memory):
            x[i] = sample
            y[i] = value
            y[i, action] = reward + discount_factor * value_max_next
            value_max_next = value.max()
        return x, y, None

class PolicyGradientLearner(ReinforcementLearner):
    def __init__(self, *args, policy_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_network_path = policy_network_path
        self.init_policy_network()

    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(
            reversed(self.memory_sample[-batch_size:]),
            reversed(self.memory_action[-batch_size:]),
            reversed(self.memory_policy[-batch_size:]),
            reversed(self.memory_reward[-batch_size:]),
        )
        x = np.zeros((batch_size, self.n_steps, self.num_features))
        y = np.full((batch_size, self.agent.NUM_ACTIONS), .5)
        for i, (sample, action, policy, reward) in enumerate(memory):
            x[i] = sample
            y[i, action] = sigmoid(delayed_reward - reward)
            y[i, 1 - action] = 1 - y[i, action]
        return x, None, y

class ActorCriticLearner(ReinforcementLearner):
    def __init__(self, *args, shared_network=None, value_network_path=None, policy_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        if shared_network is None:
            self.shared_network = Network.get_shared_network(net=self.net, n_steps=self.n_steps, input_dim=self.num_features)
        else:
            self.shared_network = shared_network
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        if self.value_network is None:
            self.init_value_network(shared_network=shared_network)
        if self.policy_network is None:
            self.init_policy_network(shared_network=shared_network)

    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(
            reversed(self.memory_sample[-batch_size:]),
            reversed(self.memory_action[-batch_size:]),
            reversed(self.memory_value[-batch_size:]),
            reversed(self.memory_policy[-batch_size:]),
            reversed(self.memory_reward[-(batch_size-1):] + [0]),
        )
        x = np.zeros((batch_size, self.n_steps, self.num_features))
        y_value = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        y_policy = np.full((batch_size, self.agent.NUM_ACTIONS), .5)
        value_max_next = 0
        for i, (sample, action, value, policy, reward) in enumerate(memory):
            x[i] = sample
            y_value[i] = value
            y_policy[i] = policy
            y_value[i, action] = reward + discount_factor * value_max_next
            a = np.argmax(y_value[i])
            v = y_value[i].max()
            y_policy[i, a] = sigmoid(v)
            value_max_next = value.max()
        return x, y_value, y_policy


class A2CLearner(ActorCriticLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(
            reversed(self.memory_sample[-batch_size:]),
            reversed(self.memory_action[-batch_size:]),
            reversed(self.memory_value[-batch_size:]),
            reversed(self.memory_policy[-batch_size:]),
            reversed(self.memory_reward[-(batch_size-1):] + [0]),
        )
        x = np.zeros((batch_size, self.n_steps, self.num_features))
        y_value = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        y_policy = np.full((batch_size, self.agent.NUM_ACTIONS), .5)
        value_max_next = 0
        for i, (sample, action, value, policy, reward) in enumerate(memory):
            x[i] = sample
            y_value[i] = value
            y_policy[i] = policy
            y_value[i, action] = reward + discount_factor * value_max_next
            advantage = value[action] - np.mean(value)
            y_policy[i, action] = sigmoid(advantage)
            value_max_next = value.max()
        return x, y_value, y_policy


class A3CLearner(A2CLearner):
    def __init__(self, *args, list_stock_code=None, list_chart_data=None, list_training_data=None,
                 list_min_trading_unit=None, list_max_trading_unit=None, **kwargs):
        super().__init__(*args, **kwargs)

        # A2CLearner 생성
        self.learners = []
        for stock_code, chart_data, training_data, min_trading_unit, max_trading_unit in zip(
                list_stock_code, list_chart_data, list_training_data,
                list_min_trading_unit, list_max_trading_unit
            ):
            learner = A2CLearner(*args, rl_method='a3c', stock_code=stock_code, chart_data=chart_data, training_data=training_data,
                    min_trading_unit=min_trading_unit, max_trading_unit=max_trading_unit, **kwargs)
            self.learners.append(learner)

    def fit(
        self, num_epoches=100, balance=10000000,
        discount_factor=1, start_epsilon=1, learning=True):
        threads = []
        for learner in self.learners:
            threads.append(threading.Thread(target=learner.fit, daemon=True, kwargs={
                'num_epoches': num_epoches, 'balance': balance,
                'discount_factor': discount_factor, 'start_epsilon': start_epsilon,
                'learning': learning
            }))
        for thread in threads:
            thread.start()
            time.sleep(1)
        for thread in threads: thread.join()
