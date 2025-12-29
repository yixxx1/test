import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime, timedelta
import warnings
import os
import glob
warnings.filterwarnings('ignore')

# 导入自定义模块
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer

class TrafficForecaster:
    """流量预测器"""
    def __init__(self, model_path, metadata_path, data_path, holiday_path, city=None):
        """
        Args:
            model_path: 模型文件路径
            metadata_path: 模型元数据路径
            data_path: 原始数据路径
            holiday_path: 节假日数据路径
            city: 城市，如果为None则从元数据中获取
        """
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.data_path = data_path
        self.holiday_path = holiday_path
        
        # 加载模型和元数据
        self.model, self.metadata = self.load_model()
        
        # 确定城市
        if city is not None:
            self.city = city
        elif 'city' in self.metadata:
            self.city = self.metadata['city']
        else:
            # 尝试从文件名中提取城市信息
            if '_A.' in model_path:
                self.city = 'A'
            elif '_B.' in model_path:
                self.city = 'B'
            elif '_C.' in model_path:
                self.city = 'C'
            else:
                self.city = 'A'  # 默认城市A
        
        self.feature_columns = self.metadata['feature_columns']
        self.scaler = self.metadata['scaler']
        
        print(f"✅ 加载模型: {self.metadata.get('model_type', 'Unknown')}")
        print(f"✅ 城市: {self.city}")
        print(f"✅ 特征数量: {len(self.feature_columns)}")
    
    def load_model(self):
        """加载模型和元数据"""
        try:
            # 先尝试用joblib加载
            model = joblib.load(self.model_path)
            metadata = joblib.load(self.metadata_path)
        except (UnicodeDecodeError, pickle.UnpicklingError):
            # 如果joblib失败，用pickle加载
            print("使用pickle加载模型...")
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            with open(self.metadata_path, 'rb') as f:
                metadata = pickle.load(f)
        
        return model, metadata
    
    def prepare_future_data(self, start_date, end_date):
        """
        准备未来时间段的数据
        Args:
            start_date: 开始日期 '2019-01-01 00:00:00'
            end_date: 结束日期 '2019-02-28 23:00:00'
        """
        print(f"为城市 {self.city} 准备未来时间段数据...")
        
        # 生成未来时间序列
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        future_dates = pd.date_range(start=start, end=end, freq='H')
        
        # 加载历史数据用于特征工程
        preprocessor = DataPreprocessor(self.data_path, self.holiday_path)
        data, holiday_data = preprocessor.load_data()
        
        # 过滤当前城市数据
        if self.city:
            data = data[data['地市'] == self.city].copy()
        
        # 预处理
        preprocessor.handle_missing_values(method='interpolate')
        preprocessor.normalize_data(method='minmax')
        processed_data = preprocessor.data
        
        # 创建特征工程器
        fe = FeatureEngineer(processed_data, holiday_data)
        fe.create_time_features()
        fe.create_holiday_features()
        fe.create_lag_features(lags=[1, 2, 3, 24, 48, 168])
        fe.create_rolling_features(windows=[3, 6, 12, 24, 72, 168])
        fe.create_advanced_features()
        fe.create_interaction_features()
        
        # 获取历史数据的特征
        X_historical, y_historical, feature_data = fe.prepare_final_features(
            target_col='流量_normalized'
        )
        
        # 创建未来数据的DataFrame
        future_df = pd.DataFrame({
            '时间': future_dates,
            '地市': self.city
        })
        
        # 为未来数据创建特征
        future_df = self._create_features_for_future(future_df, feature_data, holiday_data)
        
        print(f"城市 {self.city} 未来数据形状: {future_df.shape}")
        print(f"时间范围: {future_df['时间'].min()} 到 {future_df['时间'].max()}")
        
        return future_df
    
    def _create_features_for_future(self, future_df, historical_features, holiday_data):
        """为未来数据创建特征"""
        # 复制历史特征数据以供参考
        historical_df = historical_features.copy()
        
        # 创建时间特征
        future_df['year'] = future_df['时间'].dt.year
        future_df['month'] = future_df['时间'].dt.month
        future_df['day'] = future_df['时间'].dt.day
        future_df['hour'] = future_df['时间'].dt.hour
        future_df['dayofweek'] = future_df['时间'].dt.dayofweek
        future_df['weekofyear'] = future_df['时间'].dt.isocalendar().week
        future_df['quarter'] = future_df['时间'].dt.quarter
        
        # 时间周期特征
        future_df['sin_hour'] = np.sin(2 * np.pi * future_df['hour'] / 24)
        future_df['cos_hour'] = np.cos(2 * np.pi * future_df['hour'] / 24)
        future_df['sin_day'] = np.sin(2 * np.pi * future_df['dayofweek'] / 7)
        future_df['cos_day'] = np.cos(2 * np.pi * future_df['dayofweek'] / 7)
        future_df['sin_month'] = np.sin(2 * np.pi * future_df['month'] / 12)
        future_df['cos_month'] = np.cos(2 * np.pi * future_df['month'] / 12)
        
        # 时间标志特征
        future_df['is_weekend'] = future_df['dayofweek'].isin([5, 6]).astype(int)
        future_df['is_workday'] = (~future_df['dayofweek'].isin([5, 6])).astype(int)
        future_df['is_night'] = ((future_df['hour'] >= 0) & (future_df['hour'] <= 6)).astype(int)
        future_df['is_morning'] = ((future_df['hour'] >= 7) & (future_df['hour'] <= 12)).astype(int)
        future_df['is_afternoon'] = ((future_df['hour'] >= 13) & (future_df['hour'] <= 18)).astype(int)
        future_df['is_evening'] = ((future_df['hour'] >= 19) & (future_df['hour'] <= 23)).astype(int)
        
        # 节假日特征 - 改进版本
        from sklearn.preprocessing import LabelEncoder
        
        # 创建2019年节假日日期列表
        holiday_dates_2019 = []
        holiday_mapping = {}
        
        for _, row in holiday_data.iterrows():
            holiday_name = row['节日']
            # 只提取2019年的节假日
            start_col = '开始时间_2019'
            end_col = '结束时间_2019'
            
            if start_col in row and end_col in row and pd.notna(row[start_col]) and pd.notna(row[end_col]):
                start_date = pd.to_datetime(row[start_col])
                end_date = pd.to_datetime(row[end_col])
                
                current_date = start_date
                while current_date <= end_date:
                    holiday_dates_2019.append(current_date)
                    holiday_mapping[current_date] = holiday_name
                    current_date += timedelta(hours=1)
        
        # 标记节假日
        future_df['is_holiday'] = future_df['时间'].isin(holiday_dates_2019).astype(int)
        
        # 计算距离最近节假日的天数
        if holiday_dates_2019:
            # 提取日期（不含时间）
            future_dates_only = future_df['时间'].dt.date
            holiday_dates_only = list(set([d.date() for d in holiday_dates_2019]))
            
            # 计算每个日期到所有节假日的距离
            date_distances = {}
            for date in future_dates_only.unique():
                distances = [abs((date - h_date).days) for h_date in holiday_dates_only]
                date_distances[date] = min(distances) if distances else 0
            
            future_df['days_to_nearest_holiday'] = future_dates_only.map(date_distances)
            future_df['is_before_holiday'] = (future_df['days_to_nearest_holiday'] <= 3).astype(int)
            future_df['is_after_holiday'] = (future_df['days_to_nearest_holiday'] == 0).astype(int)
        else:
            future_df['days_to_nearest_holiday'] = 0
            future_df['is_before_holiday'] = 0
            future_df['is_after_holiday'] = 0
        
        # 节假日类型编码
        future_df['holiday_type'] = 'normal'
        for date in holiday_dates_2019:
            mask = future_df['时间'] == date
            if mask.any():
                future_df.loc[mask, 'holiday_type'] = holiday_mapping.get(date, 'holiday')
        
        le = LabelEncoder()
        future_df['holiday_encoded'] = le.fit_transform(future_df['holiday_type'])
        
        # 需要滞后特征，但未来数据没有历史流量
        # 使用历史数据的最后值作为初始滞后值
        last_historical_data = historical_df.iloc[-168*2:]  # 最后2周的数据
        
        # 为未来数据创建滞后特征占位符
        lag_features = [col for col in historical_df.columns if col.startswith('lag_')]
        for lag_feature in lag_features:
            future_df[lag_feature] = np.nan
        
        # 滑动窗口特征也类似处理
        rolling_features = [col for col in historical_df.columns if col.startswith('rolling_')]
        for rolling_feature in rolling_features:
            future_df[rolling_feature] = np.nan
        
        # 变化率特征 - 在递归预测中计算
        change_features = [col for col in historical_df.columns if col.startswith('change_') or col.startswith('pct_change_')]
        for change_feature in change_features:
            future_df[change_feature] = 0
        
        # 其他高级特征
        future_df['hour_weekday'] = future_df['hour'].astype(str) + '_' + future_df['dayofweek'].astype(str)
        future_df['hour_weekday_encoded'] = le.fit_transform(future_df['hour_weekday'])
        future_df['city_encoded'] = le.fit_transform(future_df['地市'])
        future_df['time_idx'] = np.arange(len(historical_df), len(historical_df) + len(future_df))
        
        future_df['day_of_year'] = future_df['时间'].dt.dayofyear
        future_df['sin_day_of_year'] = np.sin(2 * np.pi * future_df['day_of_year'] / 365)
        future_df['cos_day_of_year'] = np.cos(2 * np.pi * future_df['day_of_year'] / 365)
        
        # 业务特征 - 初始化为0，在递归预测中更新
        future_df['traffic_intensity'] = 0
        future_df['peak_hour_traffic'] = 0
        
        # 交互特征
        future_df['hour_x_weekday'] = future_df['hour'] * future_df['dayofweek']
        future_df['holiday_x_weekend'] = future_df['is_holiday'] * future_df['is_weekend']
        future_df['lag24_x_hour'] = 0  # 初始化为0，在递归预测中更新
        future_df['lag24_x_is_holiday'] = 0  # 初始化为0，在递归预测中更新
        future_df['rollmean24_x_hour'] = 0  # 初始化为0，在递归预测中更新
        
        # 确保所有训练时使用的特征都存在
        for col in self.feature_columns:
            if col not in future_df.columns:
                print(f"⚠️  为城市 {self.city} 添加缺失特征列: {col}")
                future_df[col] = 0
        
        return future_df
    
    def recursive_forecast(self, future_df, historical_features, steps_ahead=24):
        """
        递归预测
        Args:
            future_df: 未来数据DataFrame
            historical_features: 历史特征数据
            steps_ahead: 每次预测的步长
        """
        print(f"为城市 {self.city} 执行递归预测...")
        
        # 获取历史数据的最后部分用于初始化
        historical_df = historical_features.copy()
        
        # 获取历史流量数据的最后值
        if '流量_normalized' in historical_df.columns:
            last_flow_values = historical_df['流量_normalized'].values[-168*2:]  # 最后2周的数据
        else:
            last_flow_values = np.zeros(168*2)
        
        # 初始化预测结果列表
        predictions = []
        
        # 维护最近预测值的列表，用于计算滚动特征
        recent_predictions = []
        
        # 对于每个时间点进行预测
        for i in range(len(future_df)):
            if i % 100 == 0:
                print(f"  城市 {self.city}: 已预测 {i}/{len(future_df)} 小时")
            
            # 准备当前时间点的特征
            current_features = future_df.iloc[i:i+1].copy()
            
            # 如果有滞后特征，需要用最新的预测值更新
            if i > 0 and len(predictions) > 0:
                # 更新滞后特征
                for lag in [1, 2, 3, 24, 48, 168]:
                    lag_col = f'lag_{lag}h'
                    if lag_col in self.feature_columns:
                        if i >= lag:
                            # 使用之前的预测值
                            current_features[lag_col] = predictions[i-lag]
                        else:
                            # 使用历史数据
                            if len(last_flow_values) >= lag - i:
                                current_features[lag_col] = last_flow_values[-(lag-i)]
            
            # 更新变化率特征
            if i > 0 and 'lag_1h' in self.feature_columns:
                current_lag_1h = current_features.get('lag_1h', 0).values[0] if hasattr(current_features.get('lag_1h', 0), 'values') else current_features.get('lag_1h', 0)
                if i > 0 and len(predictions) > 0:
                    # 更新change_1h和pct_change_1h
                    current_features['change_1h'] = predictions[-1] - current_lag_1h
                    current_features['pct_change_1h'] = current_features['change_1h'] / (current_lag_1h + 1e-8)
            
            if i >= 24 and 'lag_24h' in self.feature_columns:
                current_lag_24h = current_features.get('lag_24h', 0).values[0] if hasattr(current_features.get('lag_24h', 0), 'values') else current_features.get('lag_24h', 0)
                if i >= 24 and len(predictions) >= 24:
                    # 更新change_24h和pct_change_24h
                    current_features['change_24h'] = predictions[-1] - current_lag_24h
                    current_features['pct_change_24h'] = current_features['change_24h'] / (current_lag_24h + 1e-8)
            
            if i >= 168 and 'lag_168h' in self.feature_columns:
                current_lag_168h = current_features.get('lag_168h', 0).values[0] if hasattr(current_features.get('lag_168h', 0), 'values') else current_features.get('lag_168h', 0)
                if i >= 168 and len(predictions) >= 168:
                    # 更新change_168h和pct_change_168h
                    current_features['change_168h'] = predictions[-1] - current_lag_168h
                    current_features['pct_change_168h'] = current_features['change_168h'] / (current_lag_168h + 1e-8)
            
            # 更新滑动窗口特征
            recent_predictions.append(0)  # 占位符，将在预测后更新
            
            for window in [3, 6, 12, 24, 72, 168]:
                mean_col = f'rolling_mean_{window}h'
                std_col = f'rolling_std_{window}h'
                max_col = f'rolling_max_{window}h'
                min_col = f'rolling_min_{window}h'
                median_col = f'rolling_median_{window}h'
                
                if i >= window:
                    # 使用最近的预测值计算滚动统计
                    recent_values = predictions[max(0, i-window):i]
                    if len(recent_values) > 0:
                        if mean_col in self.feature_columns:
                            current_features[mean_col] = np.mean(recent_values)
                        if std_col in self.feature_columns:
                            current_features[std_col] = np.std(recent_values) if len(recent_values) > 1 else 0
                        if max_col in self.feature_columns:
                            current_features[max_col] = np.max(recent_values)
                        if min_col in self.feature_columns:
                            current_features[min_col] = np.min(recent_values)
                        if median_col in self.feature_columns:
                            current_features[median_col] = np.median(recent_values)
            
            # 更新交互特征
            if 'lag24_x_hour' in self.feature_columns and 'lag_24h' in current_features.columns:
                current_features['lag24_x_hour'] = current_features['lag_24h'] * current_features['hour']
            
            if 'lag24_x_is_holiday' in self.feature_columns and 'lag_24h' in current_features.columns:
                current_features['lag24_x_is_holiday'] = current_features['lag_24h'] * current_features['is_holiday']
            
            if 'rollmean24_x_hour' in self.feature_columns and 'rolling_mean_24h' in current_features.columns:
                current_features['rollmean24_x_hour'] = current_features['rolling_mean_24h'] * current_features['hour']
            
            # 更新业务特征
            if 'traffic_intensity' in self.feature_columns:
                current_features['traffic_intensity'] = 0  # 将在预测后更新
            
            if 'peak_hour_traffic' in self.feature_columns:
                current_features['peak_hour_traffic'] = 0  # 将在预测后更新
            
            # 确保所有特征都存在
            for col in self.feature_columns:
                if col not in current_features.columns:
                    current_features[col] = 0
            
            # 选择模型需要的特征
            features_for_prediction = current_features[self.feature_columns].fillna(0)
            
            # 特征缩放
            features_scaled = self.scaler.transform(features_for_prediction)
            
            # 预测
            try:
                pred = self.model.predict(features_scaled)[0]
                predictions.append(pred)
            except Exception as e:
                print(f"城市 {self.city} 预测失败: {e}")
                # 如果预测失败，使用历史均值
                predictions.append(historical_df['流量_normalized'].mean())
            
            # 更新业务特征
            if 'traffic_intensity' in self.feature_columns and i < len(predictions):
                predictions[-1] = predictions[-1] * current_features['is_workday'].values[0] if current_features['is_workday'].values[0] else predictions[-1]
            
            if 'peak_hour_traffic' in self.feature_columns and i < len(predictions):
                hour = current_features['hour'].values[0]
                if 18 <= hour <= 22:
                    predictions[-1] = predictions[-1] * 1.2  # 高峰时段增加20%
        
        return np.array(predictions)
    
    def forecast(self, start_date, end_date):
        """
        执行预测
        Args:
            start_date: 开始日期
            end_date: 结束日期
        Returns:
            预测结果的DataFrame
        """
        print("="*50)
        print(f"开始为城市 {self.city} 预测流量")
        print("="*50)
        
        # 1. 准备未来数据
        future_df = self.prepare_future_data(start_date, end_date)
        
        # 2. 加载历史数据用于特征参考
        preprocessor = DataPreprocessor(self.data_path, self.holiday_path)
        data, holiday_data = preprocessor.load_data()
        
        if self.city:
            data = data[data['地市'] == self.city].copy()
        
        preprocessor.handle_missing_values(method='interpolate')
        preprocessor.normalize_data(method='minmax')
        processed_data = preprocessor.data
        
        fe = FeatureEngineer(processed_data, holiday_data)
        fe.create_time_features()
        fe.create_holiday_features()
        fe.create_lag_features(lags=[1, 2, 3, 24, 48, 168])
        fe.create_rolling_features(windows=[3, 6, 12, 24, 72, 168])
        fe.create_advanced_features()
        fe.create_interaction_features()
        
        X_historical, y_historical, historical_features = fe.prepare_final_features(
            target_col='流量_normalized'
        )
        
        # 3. 递归预测
        predictions_normalized = self.recursive_forecast(future_df, historical_features)
        
        # 4. 反归一化预测结果
        predictions_original = self._denormalize_predictions(predictions_normalized)
        
        # 5. 创建结果DataFrame
        result_df = pd.DataFrame({
            '时间': future_df['时间'],
            '地市': self.city,
            '流量': predictions_original
        })
        
        # 添加序号列
        result_df.insert(0, '序号', range(1, len(result_df) + 1))
        
        print(f"✅ 城市 {self.city} 预测完成！共预测 {len(result_df)} 小时数据")
        print(f"📊 城市 {self.city} 预测流量范围: {result_df['流量'].min():.2f} - {result_df['流量'].max():.2f}")
        
        return result_df
    
    def _denormalize_predictions(self, predictions_normalized):
        """反归一化预测结果"""
        # 从历史数据中获取流量范围
        preprocessor = DataPreprocessor(self.data_path, self.holiday_path)
        data, _ = preprocessor.load_data()
        
        if self.city:
            data = data[data['地市'] == self.city].copy()
        
        min_flow = data['流量'].min()
        max_flow = data['流量'].max()
        
        predictions_original = predictions_normalized * (max_flow - min_flow) + min_flow
        
        return predictions_original

class MultiCityForecaster:
    """多城市预测器"""
    def __init__(self, model_dir='models', data_path='data/data.csv', holiday_path='data/holiday_dates.csv'):
        """
        Args:
            model_dir: 模型文件目录
            data_path: 原始数据路径
            holiday_path: 节假日数据路径
        """
        self.model_dir = model_dir
        self.data_path = data_path
        self.holiday_path = holiday_path
        self.cities = ['A', 'B', 'C']
        
        # 检查模型文件
        self.available_models = self._find_available_models()
        
    def _find_available_models(self):
        """查找可用的模型文件"""
        available_models = {}
        
        for city in self.cities:
            # 查找模型文件
            model_pattern = os.path.join(self.model_dir, f'*_{city}.pkl')
            model_files = glob.glob(model_pattern)
            
            # 过滤掉元数据文件
            model_files = [f for f in model_files if '_metadata' not in f]
            
            if model_files:
                # 使用第一个找到的模型文件
                model_path = model_files[0]
                metadata_path = model_path.replace('.pkl', '_metadata.pkl')
                
                if os.path.exists(metadata_path):
                    available_models[city] = {
                        'model_path': model_path,
                        'metadata_path': metadata_path
                    }
                    print(f"✅ 找到城市 {city} 的模型: {os.path.basename(model_path)}")
                else:
                    print(f"⚠️  城市 {city} 的元数据文件不存在: {metadata_path}")
            else:
                print(f"⚠️  未找到城市 {city} 的模型文件")
        
        return available_models
    
    def forecast_all_cities(self, start_date, end_date):
        """
        预测所有城市的流量
        Args:
            start_date: 开始日期
            end_date: 结束日期
        Returns:
            字典：城市->预测结果的DataFrame
        """
        print("="*50)
        print("开始多城市流量预测")
        print("="*50)
        
        city_predictions = {}
        
        for city in self.cities:
            if city in self.available_models:
                print(f"\n处理城市 {city}...")
                
                try:
                    # 创建预测器
                    forecaster = TrafficForecaster(
                        model_path=self.available_models[city]['model_path'],
                        metadata_path=self.available_models[city]['metadata_path'],
                        data_path=self.data_path,
                        holiday_path=self.holiday_path,
                        city=city
                    )
                    
                    # 执行预测
                    city_prediction = forecaster.forecast(start_date, end_date)
                    city_predictions[city] = city_prediction
                    
                except Exception as e:
                    print(f"❌ 城市 {city} 预测失败: {e}")
            else:
                print(f"❌ 城市 {city} 没有可用的模型，跳过预测")
        
        if not city_predictions:
            print("❌ 没有成功预测任何城市")
            return None
        
        return city_predictions
    
    def save_city_predictions(self, city_predictions, output_dir='results'):
        """保存各城市的预测结果"""
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        for city, predictions_df in city_predictions.items():
            # 生成文件名
            filename = f'prediction_city_{city}.csv'
            filepath = os.path.join(output_dir, filename)
            
            # 保存为CSV
            predictions_df.to_csv(filepath, index=False)
            saved_files[city] = filepath
            
            print(f"✅ 城市 {city} 预测结果已保存到: {filepath}")
            print(f"   预测数量: {len(predictions_df)} 小时数据")
            print(f"   时间范围: {predictions_df['时间'].min()} 到 {predictions_df['时间'].max()}")
            print(f"   流量范围: {predictions_df['流量'].min():.2f} - {predictions_df['流量'].max():.2f}")
        
        # 可选：保存合并的文件
        combined_df = pd.concat(city_predictions.values(), ignore_index=True)
        combined_df = combined_df.sort_values(['时间', '地市']).reset_index(drop=True)
        
        combined_filename = 'multicity_predictions_combined.csv'
        combined_filepath = os.path.join(output_dir, combined_filename)
        combined_df.to_csv(combined_filepath, index=False)
        
        print(f"\n✅ 合并预测结果已保存到: {combined_filepath}")
        print(f"   总预测数量: {len(combined_df)} 小时数据")
        print(f"   包含城市: {', '.join(sorted(city_predictions.keys()))}")
        
        return saved_files, combined_filepath
    
    def visualize_predictions(self, city_predictions):
        """可视化预测结果"""
        import matplotlib.pyplot as plt
        
        if not city_predictions:
            print("❌ 没有预测数据可可视化")
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 各城市预测流量时间序列
        for city, predictions_df in city_predictions.items():
            axes[0, 0].plot(predictions_df['时间'], predictions_df['流量'], label=f'城市 {city}', alpha=0.7)
        
        axes[0, 0].set_title('各城市预测流量时间序列')
        axes[0, 0].set_xlabel('时间')
        axes[0, 0].set_ylabel('流量')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 各城市每日平均流量
        all_data = pd.concat(city_predictions.values(), ignore_index=True)
        all_data['hour'] = all_data['时间'].dt.hour
        
        for city, predictions_df in city_predictions.items():
            predictions_df['hour'] = predictions_df['时间'].dt.hour
            hourly_avg = predictions_df.groupby('hour')['流量'].mean()
            axes[0, 1].plot(hourly_avg.index, hourly_avg.values, label=f'城市 {city}', marker='o', alpha=0.7)
        
        axes[0, 1].set_title('各城市每日流量模式')
        axes[0, 1].set_xlabel('小时')
        axes[0, 1].set_ylabel('平均流量')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 各城市周流量模式
        for city, predictions_df in city_predictions.items():
            predictions_df['dayofweek'] = predictions_df['时间'].dt.dayofweek
            weekday_avg = predictions_df.groupby('dayofweek')['流量'].mean()
            axes[1, 0].plot(weekday_avg.index, weekday_avg.values, label=f'城市 {city}', marker='o', alpha=0.7)
        
        axes[1, 0].set_title('各城市周流量模式')
        axes[1, 0].set_xlabel('星期几 (0=周一)')
        axes[1, 0].set_ylabel('平均流量')
        axes[1, 0].set_xticks(range(7))
        axes[1, 0].set_xticklabels(['周一', '周二', '周三', '周四', '周五', '周六', '周日'])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 各城市预测流量分布
        for city, predictions_df in city_predictions.items():
            axes[1, 1].hist(predictions_df['流量'], bins=30, alpha=0.5, label=f'城市 {city}')
        
        axes[1, 1].set_title('各城市预测流量分布')
        axes[1, 1].set_xlabel('流量')
        axes[1, 1].set_ylabel('频率')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('A/B/C三城市网络流量预测结果 (2019年1月1日 - 2019年2月28日)', fontsize=16)
        plt.tight_layout()
        
        # 保存图表
        output_dir = 'results'
        os.makedirs(output_dir, exist_ok=True)
        chart_path = os.path.join(output_dir, 'multicity_prediction_visualization.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 可视化图表已保存到: {chart_path}")

# 预测流程示例
if __name__ == "__main__":
    # 初始化多城市预测器
    forecaster = MultiCityForecaster(
        model_dir='models',
        data_path='data/data.csv',
        holiday_path='data/holiday_dates.csv'
    )
    
    # 执行多城市预测
    start_date = '2019-01-01 00:00:00'
    end_date = '2019-02-28 23:00:00'
    
    print(f"预测时间范围: {start_date} 到 {end_date}")
    print(f"预测城市: A, B, C")
    
    # 执行预测
    city_predictions = forecaster.forecast_all_cities(start_date, end_date)
    
    if city_predictions:
        # 显示各城市预测结果预览
        print("\n各城市预测结果预览:")
        for city, predictions_df in city_predictions.items():
            print(f"\n城市 {city}:")
            print(predictions_df.head(5))
            print(f"... (共 {len(predictions_df)} 行)")
        
        # 保存各城市的预测结果
        print("\n" + "="*50)
        print("保存各城市预测结果")
        print("="*50)
        
        saved_files, combined_filepath = forecaster.save_city_predictions(city_predictions)
        
        # 统计信息
        print("\n预测结果统计:")
        for city, filepath in saved_files.items():
            predictions_df = city_predictions[city]
            print(f"城市 {city}:")
            print(f"  文件: {os.path.basename(filepath)}")
            print(f"  预测数量: {len(predictions_df)} 小时数据")
            print(f"  平均流量: {predictions_df['流量'].mean():.2f}")
            print(f"  最小流量: {predictions_df['流量'].min():.2f}")
            print(f"  最大流量: {predictions_df['流量'].max():.2f}")
        
        # 可视化预测结果
        print("\n" + "="*50)
        print("生成可视化图表")
        print("="*50)
        
        forecaster.visualize_predictions(city_predictions)
        
        print("\n" + "="*50)
        print("多城市预测完成!")
        print("="*50)
        print("生成的文件:")
        for city, filepath in saved_files.items():
            print(f"  城市 {city}: {filepath}")
        print(f"  合并文件: {combined_filepath}")
    else:
        print("\n❌ 预测失败，请检查模型文件和数据!")