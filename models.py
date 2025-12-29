import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

# 传统机器学习模型
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# 时间序列模型
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

# 树模型
import xgboost as xgb
import lightgbm as lgb

# 深度学习模型
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import inspect
class BaseModel:
    """基础模型类"""
    def __init__(self, model_name, model_params=None):
        self.model_name = model_name
        self.model_params = model_params or {}
        self.model = None
        self.scaler = None
        self.feature_importance = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型"""
        raise NotImplementedError
    
    def predict(self, X):
        """预测"""
        raise NotImplementedError
    
    def evaluate(self, X_test, y_test):
        """评估模型"""
        y_pred = self.predict(X_test)
        
        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred),
            'MAPE': self._calculate_mape(y_test, y_pred)
        }
        
        return metrics, y_pred
    
    def _calculate_mape(self, y_true, y_pred):
        """计算MAPE"""
        mask = y_true != 0
        if mask.any():
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return np.nan

class TraditionalModel(BaseModel):
    """传统机器学习模型"""
    def __init__(self, model_type='random_forest', **kwargs):
        super().__init__(model_type, kwargs)
        self.model_type = model_type
        
        # 根据类型初始化模型
        if model_type == 'linear_regression':
            self.model = LinearRegression(**self._filter_params(LinearRegression, kwargs))
        elif model_type == 'ridge':
            self.model = Ridge(**self._filter_params(Ridge, kwargs))
        elif model_type == 'lasso':
            self.model = Lasso(**self._filter_params(Lasso, kwargs))
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(**self._filter_params(RandomForestRegressor, kwargs))
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(**self._filter_params(GradientBoostingRegressor, kwargs))
        elif model_type == 'svr':
            self.model = SVR(**self._filter_params(SVR, kwargs))
        elif model_type == 'knn':
            self.model = KNeighborsRegressor(**self._filter_params(KNeighborsRegressor, kwargs))
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
    
    def _filter_params(self, model_class, params):
    """过滤出模型类支持的参数"""
    # 获取模型类的构造函数参数
    try:
        model_params = inspect.signature(model_class.__init__).parameters
        supported_params = {}
        for param_name in params:
            if param_name in model_params:
                supported_params[param_name] = params[param_name]
            else:
                # 输出警告但不中断程序
                print(f"警告: 参数 '{param_name}' 不是 {model_class.__name__} 支持的参数，已忽略")
        return supported_params
    except Exception as e:
        print(f"获取模型参数时出错: {e}")
        return {}  # 返回空字典，使用默认参数    
    def predict(self, X):
        """预测"""
        return self.model.predict(X)
class TreeModel(BaseModel):
    """树模型（XGBoost, LightGBM）"""
    def __init__(self, model_type='xgboost', **kwargs):
        super().__init__(model_type, kwargs)
        self.model_type = model_type
        
        # 设置默认参数
        default_params = {
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'lightgbm': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbose': -1  # LightGBM 的静默设置
            }
        }
        
        # 合并参数
        params = default_params.get(model_type, {}).copy()
        params.update(kwargs)
        
        if model_type == 'xgboost':
            self.model = xgb.XGBRegressor(**params)
        elif model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(**params)
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型"""
        print(f"训练{self.model_name}模型...")
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            
            # 根据模型类型调整 fit 参数
            if self.model_type == 'xgboost':
                # XGBoost 支持 verbose 参数
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    verbose=False
                )
            else:  # lightgbm
                # LightGBM 不支持 verbose 参数
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set
                )
        else:
            self.model.fit(X_train, y_train)
        
        # 获取特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self
    
    def predict(self, X):
        """预测"""
        return self.model.predict(X)
class LSTMModel(BaseModel):
    """LSTM模型"""
    def __init__(self, sequence_length=24, **kwargs):
        super().__init__('lstm', kwargs)
        self.sequence_length = sequence_length
        self.model = self._build_model(**kwargs)
    
    def _build_model(self, lstm_units=64, dropout_rate=0.2, dense_units=32):
        """构建LSTM模型"""
        model = Sequential([
            Input(shape=(self.sequence_length, 1)),
            LSTM(lstm_units, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(lstm_units // 2, return_sequences=False),
            Dropout(dropout_rate),
            Dense(dense_units, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_sequences(self, X, y):
        """创建时间序列数据"""
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i+self.sequence_length])
            y_seq.append(y[i+self.sequence_length])
        return np.array(X_seq), np.array(y_seq)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练LSTM模型"""
        print("训练LSTM模型...")
        
        # 准备数据
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        ]
        
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.create_sequences(X_val, y_val)
            history = self.model.fit(
                X_train_seq, y_train_seq,
                validation_data=(X_val_seq, y_val_seq),
                epochs=50,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
        else:
            history = self.model.fit(
                X_train_seq, y_train_seq,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
        
        self.history = history
        return self
    
    def predict(self, X):
        """预测"""
        # 创建序列
        if len(X.shape) == 1:
            X = X.values.reshape(-1, 1)
        
        # 确保有足够的序列长度
        if len(X) < self.sequence_length:
            raise ValueError(f"输入数据长度({len(X)})小于序列长度({self.sequence_length})")
        
        # 取最后sequence_length个数据点
        X_seq = X[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        return self.model.predict(X_seq)[0][0]

class ARIMAModel(BaseModel):
    """ARIMA模型"""
    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
        super().__init__('arima')
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练ARIMA模型"""
        print("训练ARIMA模型...")
        
        # ARIMA需要单变量时间序列
        if len(X_train.shape) > 1:
            # 如果有多特征，只使用目标序列
            self.model = ARIMA(y_train, order=self.order)
        else:
            self.model = ARIMA(X_train, order=self.order)
        
        self.fitted_model = self.model.fit()
        return self
    
    def predict(self, X, steps=1):
        """预测"""
        return self.fitted_model.forecast(steps=steps)

class ProphetModel(BaseModel):
    """Prophet模型"""
    def __init__(self, **kwargs):
        super().__init__('prophet')
        self.model = Prophet(**kwargs)
    
    def prepare_data(self, X, y):
        """准备Prophet格式数据"""
        if isinstance(X, pd.DataFrame) and '时间' in X.columns:
            df = pd.DataFrame({
                'ds': X['时间'],
                'y': y
            })
        else:
            # 假设X包含时间信息
            df = pd.DataFrame({
                'ds': pd.date_range(start='2017-01-01', periods=len(y), freq='H'),
                'y': y
            })
        return df
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练Prophet模型"""
        print("训练Prophet模型...")
        
        df_train = self.prepare_data(X_train, y_train)
        self.model.fit(df_train)
        
        return self
    
    def predict(self, X, periods=24):
        """预测"""
        future = self.model.make_future_dataframe(periods=periods, freq='H')
        forecast = self.model.predict(future)
        return forecast['yhat'].values[-periods:]

class ModelEnsemble:
    """模型集成"""
    def __init__(self, models, weights=None):
        """
        Args:
            models: 模型列表
            weights: 模型权重，None则等权重
        """
        self.models = models
        self.weights = weights if weights else [1/len(models)] * len(models)
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练所有模型"""
        for model in self.models:
            model.train(X_train, y_train, X_val, y_val)
        return self
    
    def predict(self, X):
        """集成预测"""
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            if isinstance(pred, (int, float, np.ndarray)):
                predictions.append(pred)
        
        # 加权平均
        if len(predictions) > 0:
            weighted_sum = 0
            total_weight = 0
            
            for i, pred in enumerate(predictions):
                weight = self.weights[i]
                weighted_sum += pred * weight
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else np.mean(predictions)
        return None

class ModelComparator:
    """模型比较器"""
    def __init__(self, models_dict):
        """
        Args:
            models_dict: {模型名: 模型实例} 字典
        """
        self.models_dict = models_dict
        self.results     = {}
        
def compare(self, X_train, y_train, X_test, y_test, cv_splits=5):
    """比较模型性能"""
    print("="*50)
    print("模型比较")
    print("="*50)
    
    # 交叉验证
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    
    for name, model in self.models_dict.items():
        print(f"\n评估模型: {name}")
        
        cv_scores = {'MAE': [], 'RMSE': [], 'R2': []}
        
        # 交叉验证
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # 使用 _clone_model 方法创建模型实例
            model_instance = self._clone_model(model)
            model_instance.train(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
            
            # 验证
            metrics, _ = model_instance.evaluate(X_fold_val, y_fold_val)
            
            for metric_name, value in metrics.items():
                if metric_name in cv_scores:
                    cv_scores[metric_name].append(value)
        
        # 计算平均CV分数
        avg_scores = {metric: np.mean(scores) for metric, scores in cv_scores.items()}
        
        # 最终训练和测试
        model.train(X_train, y_train)
        test_metrics, y_pred = model.evaluate(X_test, y_test)
        
        self.results[name] = {
            'cv_scores': avg_scores,
            'test_metrics': test_metrics,
            'predictions': y_pred,
            'model': model
        }
        
        print(f"交叉验证结果:")
        for metric, score in avg_scores.items():
            print(f"  {metric}: {score:.4f}")
        
        print(f"测试集结果:")
        for metric, score in test_metrics.items():
                print(f"  {medef _clone_model(self, model):
    """克隆模型实例"""
    # 获取模型类型和参数
    model_params = model.model_params if hasattr(model, 'model_params') else {}
    
    # 根据模型类型创建新实例
    if isinstance(model, TraditionalModel):
        # TraditionalModel需要model_type参数
        model_type = model.model_type if hasattr(model, 'model_type') else 'random_forest'
        return TraditionalModel(model_type=model_type, **model_params)
    elif isinstance(model, TreeModel):
        # TreeModel需要model_type参数
        model_type = model.model_type if hasattr(model, 'model_type') else 'xgboost'
        return TreeModel(model_type=model_type, **model_params)
    elif isinstance(model, LSTMModel):
        # LSTMModel需要sequence_length参数
        sequence_length = model.sequence_length if hasattr(model, 'sequence_length') else 24
        return LSTMModel(sequence_length=sequence_length, **model_params)
    elif isinstance(model, ARIMAModel):
        # ARIMAModel需要order和seasonal_order参数
        order = model.order if hasattr(model, 'order') else (1, 1, 1)
        seasonal_order = model.seasonal_order if hasattr(model, 'seasonal_order') else (0, 0, 0, 0)
        return ARIMAModel(order=order, seasonal_order=seasonal_order)
    elif isinstance(model, ProphetModel):
        return ProphetModel(**model_params)
    else:
        # 默认使用类型和参数创建
        return type(model)(**model_params)
            return type(model)(**model_params)
    
    def plot_comparison(self):
        """可视化模型比较结果"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 模型性能比较（条形图）
        model_names = list(self.results.keys())
        metrics_to_plot = ['RMSE', 'MAE', 'R2', 'MAPE']
        
        for idx, metric in enumerate(metrics_to_plot):
            row = idx // 2
            col = idx % 2
            
            scores = []
            for name in model_names:
                if metric in self.results[name]['test_metrics']:
                    scores.append(self.results[name]['test_metrics'][metric])
                else:
                    scores.append(0)
            
            axes[row, col].bar(model_names, scores)
            axes[row, col].set_title(f'{metric}比较')
            axes[row, col].set_ylabel(metric)
            axes[row, col].tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for i, v in enumerate(scores):
                axes[row, col].text(i, v, f'{v:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_best_model(self, metric='RMSE', higher_is_better=False):
        """获取最佳模型"""
        best_score = float('inf') if not higher_is_better else float('-inf')
        best_model = None
        best_name = None
        
        for name, result in self.results.items():
            if metric in result['test_metrics']:
                score = result['test_metrics'][metric]
                
                if (higher_is_better and score > best_score) or \
                   (not higher_is_better and score < best_score):
                    best_score = score
                    best_model = result['model']
                        best_name = name
        
        return best_name, bes  t_model, best_score

# 模型 工厂函
def create_model(model_type, **kwargs):
    """创建模型实例的工厂函数"""
    model_registry = {
        'linear_regression': TraditionalModel,
        'ridge': TraditionalModel,
        'lasso': TraditionalModel,
        'random_forest': TraditionalModel,
        'gradient_boosting': TraditionalModel,
        'svr': TraditionalModel,
        'knn': TraditionalModel,
        'xgboost': TreeModel,
        'lightgbm': TreeModel,
        'lstm': LSTMModel,
        'arima': ARIMAModel,
        'prophet': ProphetModel
    }
    
    if model_type not in model_registry:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    return model_registry[model_type](model_type=model_type, **kwargs)

# 使用示例
if __name__ == "__main__":
    # 加载特征数据
    feature_data = pd.read_csv('data/features_data.csv')
    
    # 假设我们已经准备好了训练和测试数据
    X = feature_data.drop(['流量_normalized', '时间', '地市'], axis=1)
    y = feature_data['流量_normalized']
    
    # 划分训练测试集（按时间）
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 创建多个模型
    models = {
        '随机森林': create_model('random_forest', n_estimators=100, max_depth=10, random_state=42),
        'XGBoost': create_model('xgboost', n_estimators=100, max_depth=6, learning_rate=0.1),
        'LightGBM': create_model('lightgbm', n_estimators=100, max_depth=6, learning_rate=0.1),
        'LSTM': create_model('lstm', sequence_length=24, lstm_units=64),
    }
    
    # 比较模型
    comparator = ModelComparator(models)
    results = comparator.compare(X_train, y_train, X_test, y_test, cv_splits=3)
    
    # 可视化比较结果
    comparator.plot_comparison()
    
    # 获取最佳模型
    best_name, best_model, best_score = comparator.get_best_model(metric='RMSE')
    print(f"\n最佳模型: {best_name}, RMSE: {best_score:.4f}")
    
    # 保存最佳模型
    import joblib
    joblib.dump(best_model, f'models/best_model_{best_name}.pkl')
    print(f"最佳模型已保存到: models/best_model_{best_name}.pkl")