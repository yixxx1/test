import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder

class FeatureEngineer:
    def __init__(self, data, holiday_data):
        """
        初始化特征工程师
        Args:
            data: 预处理后的数据
            holiday_data: 节假日数据
        """
        self.data = data.copy()
        self.holiday_data = holiday_data
        self.feature_data = None
        
    def create_time_features(self):
        """创建时间特征"""
        print("创建时间特征...")
        
        # 基本时间特征
        self.data['year'] = self.data['时间'].dt.year
        self.data['month'] = self.data['时间'].dt.month
        self.data['day'] = self.data['时间'].dt.day
        self.data['hour'] = self.data['时间'].dt.hour
        self.data['dayofweek'] = self.data['时间'].dt.dayofweek  # 周一=0, 周日=6
        self.data['weekofyear'] = self.data['时间'].dt.isocalendar().week
        self.data['quarter'] = self.data['时间'].dt.quarter
        
        # 时间周期特征
        self.data['sin_hour'] = np.sin(2 * np.pi * self.data['hour'] / 24)
        self.data['cos_hour'] = np.cos(2 * np.pi * self.data['hour'] / 24)
        self.data['sin_day'] = np.sin(2 * np.pi * self.data['dayofweek'] / 7)
        self.data['cos_day'] = np.cos(2 * np.pi * self.data['dayofweek'] / 7)
        self.data['sin_month'] = np.sin(2 * np.pi * self.data['month'] / 12)
        self.data['cos_month'] = np.cos(2 * np.pi * self.data['month'] / 12)
        
        # 时间标志特征
        self.data['is_weekend'] = self.data['dayofweek'].isin([5, 6]).astype(int)
        self.data['is_workday'] = (~self.data['dayofweek'].isin([5, 6])).astype(int)
        self.data['is_night'] = ((self.data['hour'] >= 0) & (self.data['hour'] <= 6)).astype(int)
        self.data['is_morning'] = ((self.data['hour'] >= 7) & (self.data['hour'] <= 12)).astype(int)
        self.data['is_afternoon'] = ((self.data['hour'] >= 13) & (self.data['hour'] <= 18)).astype(int)
        self.data['is_evening'] = ((self.data['hour'] >= 19) & (self.data['hour'] <= 23)).astype(int)
        
        print(f"创建的时间特征: {[col for col in self.data.columns if col not in ['时间', '地市', '流量']]}")
        return self.data
    
    def create_holiday_features(self):
        """创建节日特征"""
        print("创建节日特征...")
        
        # 初始化节假日特征列
        self.data['is_holiday'] = 0
        
        # 从节假日数据中提取所有节假日日期
        holiday_dates = []
        holiday_mapping = {}
        
        # 动态确定需要处理的年份（基于数据中的年份）
        years_in_data = self.data['时间'].dt.year.unique()
        print(f"数据中的年份: {sorted(years_in_data)}")
        
        for _, row in self.holiday_data.iterrows():
            holiday_name = row['节日']
            
            # 处理每个年份
            for year in sorted(years_in_data):
                start_col = f'开始时间_{year}'
                end_col = f'结束时间_{year}'
                
                # 检查列是否存在
                if start_col in row.index and end_col in row.index:
                    start_val = row[start_col]
                    end_val = row[end_col]
                    
                    # 检查是否有有效值
                    if pd.notna(start_val) and pd.notna(end_val):
                        try:
                            start_date = pd.to_datetime(start_val)
                            end_date = pd.to_datetime(end_val)
                            
                            # 生成节假日期间的每一个小时
                            current_date = start_date
                            while current_date <= end_date:
                                holiday_dates.append(current_date)
                                holiday_mapping[current_date] = holiday_name
                                current_date += timedelta(hours=1)  # 小时级数据
                        except Exception as e:
                            print(f"处理节假日 {holiday_name} 在 {year} 年时出错: {e}")
                            continue
        
        if not holiday_dates:
            print("警告: 未找到节假日数据，节假日特征将为空")
            # 创建空列并返回
            self.data['holiday_type'] = 'normal'
            le = LabelEncoder()
            self.data['holiday_encoded'] = le.fit_transform(self.data['holiday_type'])
            self.data['days_to_nearest_holiday'] = 0
            return self.data
        
        # 创建节假日DataFrame
        holiday_df = pd.DataFrame({
            '时间': holiday_dates,
            'holiday_type_temp': [holiday_mapping[d] for d in holiday_dates]  # 使用临时列名
        })
        
        # 标记节假日
        holiday_times = set(holiday_df['时间'])
        self.data['is_holiday'] = self.data['时间'].isin(holiday_times).astype(int)
        
        # 合并节假日类型 - 使用临时列名避免冲突
        self.data = pd.merge(self.data, holiday_df, on='时间', how='left', suffixes=('', '_holiday'))
        
        # 重命名列并填充缺失值
        self.data['holiday_type'] = self.data['holiday_type_temp'].fillna('normal')
        
        # 删除临时列
        if 'holiday_type_temp' in self.data.columns:
            self.data = self.data.drop('holiday_type_temp', axis=1)
        
        # 创建节假日编码
        le = LabelEncoder()
        self.data['holiday_encoded'] = le.fit_transform(self.data['holiday_type'])
        
        # 计算距离节假日的天数
        # 首先提取日期（不含时间）用于计算
        self.data['date_only'] = self.data['时间'].dt.date
        
        # 获取所有节假日日期（不含时间）
        holiday_dates_only = list(set([d.date() for d in holiday_dates]))
        
        if holiday_dates_only:
            # 对于数据中的每个日期，计算到最近节假日的距离
            date_distances = {}
            for date in self.data['date_only'].unique():
                # 计算到所有节假日的距离，取最小值
                distances = [abs((date - h_date).days) for h_date in holiday_dates_only]
                date_distances[date] = min(distances) if distances else 0
            
            self.data['days_to_nearest_holiday'] = self.data['date_only'].map(date_distances)
        else:
            self.data['days_to_nearest_holiday'] = 0
        
        # 删除临时列
        if 'date_only' in self.data.columns:
            self.data = self.data.drop('date_only', axis=1)
        
        # 添加节假日前后特征
        self.data['is_before_holiday'] = (self.data['days_to_nearest_holiday'] <= 3).astype(int)
        self.data['is_after_holiday'] = (self.data['days_to_nearest_holiday'] == 0).astype(int)
        
        print(f"节假日特征创建完成，共找到 {len(holiday_dates)} 小时级的节假日数据点")
        print(f"节假日类型分布: {self.data['holiday_type'].value_counts().to_dict()}")
        
        return self.data
    
    def create_lag_features(self, lags=[1, 2, 3, 24, 48, 24*7, 24*30]):
        """创建滞后特征"""
        print("创建滞后特征...")
        
        target_col = '流量_normalized' if '流量_normalized' in self.data.columns else '流量'
        
        for city in self.data['地市'].unique():
            city_data = self.data[self.data['地市'] == city].copy()
            city_data = city_data.sort_values('时间')
            
            for lag in lags:
                if lag < len(city_data):
                    col_name = f'lag_{lag}h'
                    self.data.loc[self.data['地市'] == city, col_name] = city_data[target_col].shift(lag).values
        
        # 添加变化率特征
        for lag in [1, 24, 168]:  # 1小时、1天、1周的变化
            lag_col = f'lag_{lag}h'
            if lag_col in self.data.columns:
                self.data[f'change_{lag}h'] = self.data[target_col] - self.data[lag_col]
                self.data[f'pct_change_{lag}h'] = (self.data[target_col] - self.data[lag_col]) / (self.data[lag_col] + 1e-8)
        
        print(f"创建的滞后特征: {[col for col in self.data.columns if 'lag_' in col or 'change_' in col]}")
        return self.data
    
    def create_rolling_features(self, windows=[3, 6, 12, 24, 168]):
        """创建滑动窗口统计特征"""
        print("创建滑动窗口特征...")
        
        target_col = '流量_normalized' if '流量_normalized' in self.data.columns else '流量'
        
        for city in self.data['地市'].unique():
            city_mask = self.data['地市'] == city
            city_data = self.data[city_mask].sort_values('时间')
            
            for window in windows:
                if window < len(city_data):
                    # 滚动均值
                    self.data.loc[city_mask, f'rolling_mean_{window}h'] = (
                        city_data[target_col].rolling(window=window, min_periods=1).mean().values
                    )
                    
                    # 滚动标准差
                    self.data.loc[city_mask, f'rolling_std_{window}h'] = (
                        city_data[target_col].rolling(window=window, min_periods=1).std().values
                    )
                    
                    # 滚动最大值
                    self.data.loc[city_mask, f'rolling_max_{window}h'] = (
                        city_data[target_col].rolling(window=window, min_periods=1).max().values
                    )
                    
                    # 滚动最小值
                    self.data.loc[city_mask, f'rolling_min_{window}h'] = (
                        city_data[target_col].rolling(window=window, min_periods=1).min().values
                    )
                    
                    # 滚动分位数
                    self.data.loc[city_mask, f'rolling_median_{window}h'] = (
                        city_data[target_col].rolling(window=window, min_periods=1).median().values
                    )
        
        print(f"创建的滑动窗口特征: {[col for col in self.data.columns if 'rolling_' in col]}")
        return self.data
    
    def create_advanced_features(self):
        """创建高级特征"""
        print("创建高级特征...")
        
        # 星期几和小时组合特征
        self.data['hour_weekday'] = self.data['hour'].astype(str) + '_' + self.data['dayofweek'].astype(str)
        
        # 编码类别特征
        le = LabelEncoder()
        self.data['hour_weekday_encoded'] = le.fit_transform(self.data['hour_weekday'])
        
        # 城市编码
        self.data['city_encoded'] = le.fit_transform(self.data['地市'])
        
        # 时间序列趋势特征
        self.data['time_idx'] = np.arange(len(self.data))
        
        # 季节性特征（假设每年相似）
        self.data['day_of_year'] = self.data['时间'].dt.dayofyear
        self.data['sin_day_of_year'] = np.sin(2 * np.pi * self.data['day_of_year'] / 365)
        self.data['cos_day_of_year'] = np.cos(2 * np.pi * self.data['day_of_year'] / 365)
        
        # 业务特征（假设）
        self.data['traffic_intensity'] = self.data['流量_normalized'] * self.data['is_workday']
        self.data['peak_hour_traffic'] = self.data['流量_normalized'] * ((self.data['hour'] >= 18) & (self.data['hour'] <= 22)).astype(int)
        
        print("高级特征创建完成")
        return self.data
    
    def create_interaction_features(self):
        """创建交互特征"""
        print("创建交互特征...")
        
        # 小时和星期几的交互
        self.data['hour_x_weekday'] = self.data['hour'] * self.data['dayofweek']
        
        # 节假日和工作日的交互
        self.data['holiday_x_weekend'] = self.data['is_holiday'] * self.data['is_weekend']
        
        # 滞后特征和时间的交互
        if 'lag_24h' in self.data.columns:
            self.data['lag24_x_hour'] = self.data['lag_24h'] * self.data['hour']
            self.data['lag24_x_is_holiday'] = self.data['lag_24h'] * self.data['is_holiday']
        
        # 滚动特征和时间的交互
        if 'rolling_mean_24h' in self.data.columns:
            self.data['rollmean24_x_hour'] = self.data['rolling_mean_24h'] * self.data['hour']
        
        print("交互特征创建完成")
        return self.data
    
    def select_features(self, correlation_threshold=0.8):
        """
        特征选择
        Args:
            correlation_threshold: 相关性阈值，高于此值的特征会被移除
        """
        print("执行特征选择...")
        
        # 选择数值特征
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 移除目标列和时间索引
        exclude_cols = ['流量', '流量_normalized', 'time_idx', 'year', 'month', 'day']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # 计算特征相关性
        correlation_matrix = self.data[feature_cols].corr().abs()
        
        # 移除高相关特征
        upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
        
        print(f"移除的高相关特征({len(to_drop)}个): {to_drop}")
        
        # 更新数据，移除高相关特征
        self.data = self.data.drop(columns=to_drop)
        
        return self.data
    
    def prepare_final_features(self, target_col='流量_normalized'):
        """
        准备最终特征数据集
        Args:
            target_col: 目标列名
        """
        print("准备最终特征数据集...")
        
        # 移除包含NaN的行（由于滞后特征）
        original_shape = self.data.shape
        self.feature_data = self.data.dropna().copy()
        
        # 分离特征和目标
        exclude_cols = ['时间', '地市', '流量', 'holiday_type', 'hour_weekday']
        if target_col in self.feature_data.columns:
            exclude_cols.append(target_col)
        
        feature_cols = [col for col in self.feature_data.columns if col not in exclude_cols]
        X = self.feature_data[feature_cols]
        y = self.feature_data[target_col] if target_col in self.feature_data.columns else None
        
        print(f"原始数据形状: {original_shape}")
        print(f"处理后数据形状: {self.feature_data.shape}")
        print(f"特征数量: {len(feature_cols)}")
        print(f"特征列: {feature_cols}")
        
        return X, y, self.feature_data
    
    def save_features(self, output_path='data/features_data.csv'):
        """保存特征数据"""
        if self.feature_data is not None:
            self.feature_data.to_csv(output_path, index=False)
            print(f"\n特征数据已保存到: {output_path}")
            return output_path
    
    def visualize_features(self):
        """可视化特征"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        
        # 1. 小时特征可视化
        hourly_avg = self.data.groupby('hour')['流量_normalized'].mean()
        axes[0, 0].plot(hourly_avg.index, hourly_avg.values, 'b-o')
        axes[0, 0].set_title('平均流量小时分布')
        axes[0, 0].set_xlabel('小时')
        axes[0, 0].set_ylabel('平均流量')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 星期几特征可视化
        weekday_avg = self.data.groupby('dayofweek')['流量_normalized'].mean()
        axes[0, 1].plot(weekday_avg.index, weekday_avg.values, 'g-o')
        axes[0, 1].set_title('平均流量星期分布')
        axes[0, 1].set_xlabel('星期几 (0=周一)')
        axes[0, 1].set_ylabel('平均流量')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 滞后特征相关性
        lag_cols = [col for col in self.data.columns if 'lag_' in col][:5]
        if lag_cols:
            lag_corr = self.data[['流量_normalized'] + lag_cols].corr()
            sns.heatmap(lag_corr, annot=True, cmap='coolwarm', ax=axes[1, 0])
            axes[1, 0].set_title('滞后特征相关性热图')
        
        # 4. 节假日特征
        holiday_avg = self.data.groupby('is_holiday')['流量_normalized'].mean()
        axes[1, 1].bar(holiday_avg.index, holiday_avg.values)
        axes[1, 1].set_title('节假日 vs 非节假日平均流量')
        axes[1, 1].set_xlabel('是否节假日')
        axes[1, 1].set_ylabel('平均流量')
        axes[1, 1].set_xticks([0, 1])
        axes[1, 1].set_xticklabels(['非节假日', '节假日'])
        
        # 5. 特征重要性（假设有模型）
        # 这里可以添加模型特征重要性可视化
        
        # 6. 特征分布
        sample_features = [col for col in self.data.columns if col not in 
                          ['时间', '地市', '流量', '流量_normalized', 'holiday_type']][:4]
        for idx, feature in enumerate(sample_features[:4]):
            row = 2
            col = idx % 2
            axes[row, col].hist(self.data[feature].dropna(), bins=50, alpha=0.7)
            axes[row, col].set_title(f'{feature}分布')
            axes[row, col].set_xlabel('值')
            axes[row, col].set_ylabel('频率')
        
        plt.tight_layout()
        plt.savefig('features_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()

# 使用示例
if __name__ == "__main__":
    try:
        # 假设已经有预处理后的数据
        processed_data = pd.read_csv('data/processed_data.csv')
        holiday_data = pd.read_csv('data/holiday_dates.csv')
        
        # 确保时间列是datetime类型
        processed_data['时间'] = pd.to_datetime(processed_data['时间'])
        
        # 检查是否包含归一化流量列，如果没有则创建
        if '流量_normalized' not in processed_data.columns and '流量' in processed_data.columns:
            # 简单的归一化
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            processed_data['流量_normalized'] = scaler.fit_transform(
                processed_data[['流量']]
            )
        
        print(f"加载的数据形状: {processed_data.shape}")
        print(f"时间范围: {processed_data['时间'].min()} 到 {processed_data['时间'].max()}")
        
        # 初始化特征工程师
        fe = FeatureEngineer(processed_data, holiday_data)
        
        # 创建各种特征
        print("\n开始特征工程...")
        fe.create_time_features()
        fe.create_holiday_features()
        fe.create_lag_features(lags=[1, 2, 3, 24, 48, 168, 336, 720])
        fe.create_rolling_features(windows=[3, 6, 12, 24, 72, 168])
        fe.create_advanced_features()
        fe.create_interaction_features()
        
        # 特征选择
        print("\n执行特征选择...")
        fe.select_features(correlation_threshold=0.85)
        
        # 准备最终特征
        X, y, feature_data = fe.prepare_final_features(target_col='流量_normalized')
        
        # 保存特征数据
        output_path = fe.save_features('data/features_data.csv')
            
    except FileNotFoundError as e:
        print(f"文件未找到错误: {e}")
        print("请确保以下文件存在:")
        print("1. data/processed_data.csv")
        print("2. data/holiday_dates.csv")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()