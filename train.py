import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# 导入自定义模块
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from models import ModelComparator, create_model

class TrafficPredictor:
    """流量预测器"""
    def __init__(self, data_path, holiday_path, city=None):
        """
        Args:
            data_path: 数据路径
            holiday_path: 节假日数据路径
            city: 指定城市，None表示所有城市
        """
        self.data_path = data_path
        self.holiday_path = holiday_path
        self.city = city
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.scaler = None
        
    def load_and_preprocess(self):
        """加载并预处理数据"""
        print("="*50)
        print("数据加载与预处理")
        print("="*50)
        
        # 1. 数据预处理
        preprocessor = DataPreprocessor(self.data_path, self.holiday_path)
        data, holiday_data = preprocessor.load_data()
        
        # 如果指定城市，只处理该城市数据
        if self.city:
            data = data[data['地市'] == self.city].copy()
            print(f"只处理城市: {self.city}")
        
        preprocessor.handle_missing_values(method='interpolate')
        preprocessor.normalize_data(method='minmax')
        processed_data = preprocessor.data
        
        # 2. 特征工程
        print("\n" + "="*50)
        print("特征工程")
        print("="*50)
        
        fe = FeatureEngineer(processed_data, holiday_data)
        fe.create_time_features()
        fe.create_holiday_features()
        fe.create_lag_features(lags=[1, 2, 3, 24, 48, 168])
        fe.create_rolling_features(windows=[3, 6, 12, 24, 72, 168])
        fe.create_advanced_features()
        fe.create_interaction_features()
        fe.select_features(correlation_threshold=0.85)
        
        X, y, feature_data = fe.prepare_final_features(target_col='流量_normalized')
        
        # 保存特征数据
        fe.save_features(f'data/features_data_{self.city if self.city else "all"}.csv')
        
        self.data = feature_data
        self.X = X
        self.y = y
        
        print(f"最终数据集形状: X={X.shape}, y={y.shape}")
        
        return X, y
    
    def prepare_data(self, test_size=0.2, time_based=True):
        """准备训练和测试数据"""
        print("\n" + "="*50)
        print("准备训练测试数据")
        print("="*50)
        
        X, y = self.X, self.y
        
        if time_based:
            # 按时间划分（时间序列）
            split_idx = int(len(X) * (1 - test_size))
            
            self.X_train = X.iloc[:split_idx]
            self.X_test = X.iloc[split_idx:]
            self.y_train = y.iloc[:split_idx]
            self.y_test = y.iloc[split_idx:]
        else:
            # 随机划分
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        print(f"训练集: X={self.X_train.shape}, y={self.y_train.shape}")
        print(f"测试集: X={self.X_test.shape}, y={self.y_test.shape}")
        
        # 特征缩放（如果需要）
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # 保存列名
        self.feature_columns = self.X_train.columns.tolist()
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models_simple(self):
        """简化版本：分别训练每个模型"""
        print("\n" + "="*50)
        print("简化模型训练与比较")
        print("="*50)
        
        models_config = {
            '随机森林': {
                'type': 'random_forest',
                'params': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'random_state': 42
                }
            },
            'XGBoost': {
                'type': 'xgboost',
                'params': {
                    'n_estimators': 200,
                    'max_depth': 8,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                }
            },
            'LightGBM': {
                'type': 'lightgbm',
                'params': {
                    'n_estimators': 200,
                    'max_depth': 8,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'verbose': -1  # 添加这个参数
                }
            },
            '梯度提升树': {
                'type': 'gradient_boosting',
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'random_state': 42
                }
            }
        }
        
        results = {}
        best_score = float('inf')
        best_model = None
        best_name = None
        
        X_train_df = pd.DataFrame(self.X_train_scaled, columns=self.feature_columns)
        X_test_df = pd.DataFrame(self.X_test_scaled, columns=self.feature_columns)
        
        for name, config in models_config.items():
            print(f"\n训练模型: {name}")
            
            try:
                # 创建模型
                model = create_model(config['type'], **config['params'])
                
                # 训练模型
                model.train(X_train_df, self.y_train)
                
                # 评估模型
                metrics, y_pred = model.evaluate(X_test_df, self.y_test)
                
                results[name] = {
                    'model': model,
                    'metrics': metrics,
                    'predictions': y_pred
                }
                
                print(f"  测试集 RMSE: {metrics['RMSE']:.6f}, MAE: {metrics['MAE']:.6f}, R2: {metrics['R2']:.6f}")
                
                # 检查是否为最佳模型
                if metrics['RMSE'] < best_score:
                    best_score = metrics['RMSE']
                    best_model = model
                    best_name = name
                    
            except Exception as e:
                print(f"  模型 {name} 训练失败: {e}")
                continue
        
        # 可视化比较结果
        if results:
            self._plot_simple_comparison(results)
            
            print(f"\n🎯 最佳模型: {best_name}")
            print(f"📊 最佳RMSE: {best_score:.6f}")
            
            self.best_model = best_model
            self.best_model_name = best_name
            self.comparison_results = results
            
            return best_model, results
        else:
            print("\n所有模型训练都失败了！")
            return None, None
    
    def _plot_simple_comparison(self, results):
        """简单的模型比较可视化"""
        model_names = list(results.keys())
        if not model_names:
            print("没有可比较的模型结果")
            return
        
        metrics = ['RMSE', 'MAE', 'R2', 'MAPE']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        for idx, metric in enumerate(metrics):
            row = idx // 2
            col = idx % 2
            
            scores = []
            for name in model_names:
                if metric in results[name]['metrics']:
                    scores.append(results[name]['metrics'][metric])
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
        plt.savefig('model_comparison_simple.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def hyperparameter_tuning(self, model_type='xgboost'):
        """超参数调优"""
        print("\n" + "="*50)
        print("超参数调优")
        print("="*50)
        
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        
        if model_type == 'xgboost':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
            
            model = create_model('xgboost')
            base_model = model.model
            
        elif model_type == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            model = create_model('random_forest')
            base_model = model.model
        
        else:
            print(f"暂不支持 {model_type} 的超参数调优")
            return None
        
        # 使用随机搜索
        print(f"对 {model_type} 进行超参数调优...")
        
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=20,
            cv=TimeSeriesSplit(n_splits=3),
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        random_search.fit(self.X_train_scaled, self.y_train)
        
        print(f"最佳参数: {random_search.best_params_}")
        print(f"最佳分数: {-random_search.best_score_:.6f}")
        
        # 使用最佳参数重新训练模型
        best_params = random_search.best_params_
        tuned_model = create_model(model_type, **best_params)
        tuned_model.train(
            pd.DataFrame(self.X_train_scaled, columns=self.feature_columns),
            self.y_train
        )
        
        # 评估调优后的模型
        metrics, _ = tuned_model.evaluate(
            pd.DataFrame(self.X_test_scaled, columns=self.feature_columns),
            self.y_test
        )
        
        print("\n调优后模型性能:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")
        
        return tuned_model
    
    def save_model(self, model=None, model_name=None):
        """保存模型"""
        if model is None:
            model = self.best_model
        
        if model_name is None:
            model_name = self.best_model_name if hasattr(self, 'best_model_name') else 'model'
        
        # 保存模型
        model_path = f'models/{model_name}_{self.city if self.city else "all"}.pkl'
        joblib.dump(model, model_path)
        
        # 保存特征列名和缩放器
        metadata = {
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'city': self.city,
            'model_type': type(model).__name__
        }
        
        metadata_path = f'models/{model_name}_{self.city if self.city else "all"}_metadata.pkl'
        joblib.dump(metadata, metadata_path)
        
        print(f"✅ 模型已保存到: {model_path}")
        print(f"✅ 模型元数据已保存到: {metadata_path}")
        
        return model_path, metadata_path
    
    def load_model(self, model_path, metadata_path):
        """加载模型"""
        model = joblib.load(model_path)
        metadata = joblib.load(metadata_path)
        
        self.best_model = model
        self.feature_columns = metadata['feature_columns']
        self.scaler = metadata['scaler']
        self.city = metadata['city']
        
        print(f"✅ 模型已从 {model_path} 加载")
        return model, metadata
    
    def feature_importance_analysis(self):
        """特征重要性分析"""
        if self.best_model is None:
            print("请先训练模型")
            return
        
        print("\n" + "="*50)
        print("特征重要性分析")
        print("="*50)
        
        if hasattr(self.best_model, 'feature_importance'):
            importance_df = self.best_model.feature_importance
            
            if importance_df is not None:
                # 显示最重要的特征
                print("Top 20 最重要的特征:")
                print(importance_df.head(20).to_string())
                
                # 可视化特征重要性
                import matplotlib.pyplot as plt
                
                plt.figure(figsize=(12, 8))
                top_features = importance_df.head(20)
                plt.barh(range(len(top_features)), top_features['importance'])
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('重要性')
                plt.title('Top 20 特征重要性')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(f'feature_importance_{self.city if self.city else "all"}.png', 
                           dpi=300, bbox_inches='tight')
                plt.show()
            else:
                print("该模型没有特征重要性属性")
        else:
            print("该模型不支持特征重要性分析")
    
    def evaluate_custom_metrics(self):
        """计算自定义评估指标（符合任务书要求）"""
        print("\n" + "="*50)
        print("自定义指标评估")
        print("="*50)
        
        if self.best_model is None or self.X_test is None:
            print("请先训练模型")
            return
        
        # 获取预测结果
        X_test_df = pd.DataFrame(self.X_test_scaled, columns=self.feature_columns)
        metrics, y_pred = self.best_model.evaluate(X_test_df, self.y_test)
        
        # 将预测结果反归一化（如果需要）
        # 注意：这里需要实际的归一化器
        
        # 计算任务书要求的指标
        # 1. 小时粒度准确性 (权重0.7)
        # 2. 每日流量峰值准确性 (权重0.2)
        # 3. 节假日期间流量准确性 (权重0.1)
        
        # 这里需要根据任务书的具体公式计算
        print("任务书指标计算:")
        print("需要根据具体公式计算 hour_accuracy, peak_accuracy, holiday_accuracy")
        print("最终得分 = 0.7*hour_accuracy + 0.2*peak_accuracy + 0.1*holiday_accuracy")
        
        return metrics

# 训练流程示例
if __name__ == "__main__":
    # 初始化预测器
    predictor = TrafficPredictor(
        data_path='data/data.csv',
        holiday_path='data/holiday_dates.csv',
        city='A'  # 可以指定城市，或设置为None处理所有城市
    )
    
    # 1. 加载并预处理数据
    X, y = predictor.load_and_preprocess()
    
    # 2. 准备训练测试数据
    predictor.prepare_data(test_size=0.2, time_based=True)
    
    # 3. 训练并比较模型（使用简化版）
    best_model, results = predictor.train_models_simple()
    
    if best_model:
        # 4. 特征重要性分析
        predictor.feature_importance_analysis()
        
        # 5. 保存模型
        model_path, metadata_path = predictor.save_model()
        
        print("\n" + "="*50)
        print("训练完成!")
        print("="*50)
    else:
        print("\n训练失败，请检查错误信息！")