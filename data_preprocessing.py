import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
class DataPreprocessor:
    def __init__(self, data_path, holiday_path):
        """
        初始化数据预处理器
        Args:
            data_path: 流量数据文件路径
            holiday_path: 节假日数据文件路径
        """
        self.data_path = data_path
        self.holiday_path = holiday_path
        self.data = None
        self.holiday_data = None
        self.processed_data = None
        
    def load_data(self):
        """加载数据"""
        print("加载数据...")
        # 加载流量数据
        self.data = pd.read_csv(self.data_path)
        self.data['时间'] = pd.to_datetime(self.data['时间'])
        
        # 加载节假日数据
        self.holiday_data = pd.read_csv(self.holiday_path)
        print(f"流量数据形状: {self.data.shape}")
        print(f"节假日数据形状: {self.holiday_data.shape}")
        return self.data, self.holiday_data
    
    def explore_data(self):
        """探索性数据分析"""
        print("\n=== 数据探索 ===")
        
        # 基本信息
        print("数据基本信息:")
        print(self.data.info())
        print("\n数据描述统计:")
        print(self.data.describe())
        
        # 检查缺失值
        print("\n缺失值统计:")
        print(self.data.isnull().sum())
        
        # 检查重复值
        duplicates = self.data.duplicated().sum()
        print(f"\n重复行数: {duplicates}")
        
        # 检查城市分布
        print("\n城市分布:")
        print(self.data['地市'].value_counts())
        
        # 时间范围
        print(f"\n时间范围: {self.data['时间'].min()} 到 {self.data['时间'].max()}")
        
        # 可视化缺失值
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.data.isnull(), cbar=False, cmap='viridis')
        plt.title('缺失值热图')
        plt.tight_layout()
        plt.savefig('missing_values_heatmap.png', dpi=300)
        plt.show()
        
    def handle_missing_values(self, method='interpolate'):
        """
        处理缺失值
        Args:
            method: 缺失值处理方法 ('interpolate', 'ffill', 'bfill', 'mean', 'median')
        """
        print(f"\n处理缺失值，使用{method}方法...")
        
        original_missing = self.data['流量'].isnull().sum()
        
        if method == 'interpolate':
         
            self.data['流量'] = self.data.groupby('地市')['流量'].transform(
                lambda x: x.interpolate(method='linear', limit_direction='both')
            )
        elif method == 'ffill':
            # 前向填充
            self.data['流量'] = self.data.groupby('地市')['流量'].ffill()
        elif method == 'bfill':
            # 后向填充
            self.data['流量'] = self.data.groupby('地市')['流量'].bfill()
        elif method == 'mean':
            # 使用城市均值填充
            city_means = self.data.groupby('地市')['流量'].transform('mean')
            self.data['流量'] = self.data['流量'].fillna(city_means)
        elif method == 'median':
            # 使用城市中位数填充
            city_medians = self.data.groupby('地市')['流量'].transform('median')
            self.data['流量'] = self.data['流量'].fillna(city_medians)
        
        remaining_missing = self.data['流量'].isnull().sum()
        print(f"原始缺失值数: {original_missing}")
        print(f"处理后缺失值数: {remaining_missing}")
        if original_missing > 0:
            print(f"填充比例: {(original_missing - remaining_missing)/original_missing*100:.2f}%")
        
        return self.data
    
    def detect_outliers(self, method='iqr', threshold=1.5):
        """
        检测异常值
        Args:
            method: 检测方法 ('iqr', 'zscore', 'mad')
            threshold: 阈值
        """
        print(f"\n检测异常值，使用{method}方法...")
        
        outliers_info = {}
        
        for city in self.data['地市'].unique():
            city_data = self.data[self.data['地市'] == city]['流量']
            
            if method == 'iqr':
                # IQR方法
                Q1 = city_data.quantile(0.25)
                Q3 = city_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = ((city_data < lower_bound) | (city_data > upper_bound)).sum()
                
            elif method == 'zscore':
                # Z-score方法
                mean = city_data.mean()
                std = city_data.std()
                z_scores = np.abs((city_data - mean) / std)
                outliers = (z_scores > threshold).sum()
                
            elif method == 'mad':
                # MAD方法（Median Absolute Deviation）
                median = city_data.median()
                mad = np.abs(city_data - median).median()
                modified_z_scores = 0.6745 * np.abs(city_data - median) / mad
                outliers = (modified_z_scores > threshold).sum()
            
            outliers_info[city] = {
                'outliers': outliers,
                'percentage': outliers / len(city_data) * 100
            }
            
            print(f"城市 {city}: 异常值数={outliers}, 占比={outliers/len(city_data)*100:.2f}%")
        
        return outliers_info
    
    def handle_outliers(self, method='clip'):
        """
        处理异常值
        Args:
            method: 处理方法 ('clip', 'remove', 'replace_with_median')
        """
        print(f"\n处理异常值，使用{method}方法...")
        
        for city in self.data['地市'].unique():
            city_mask = self.data['地市'] == city
            city_data = self.data.loc[city_mask, '流量']
            
            # 使用IQR确定边界
            Q1 = city_data.quantile(0.25)
            Q3 = city_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            if method == 'clip':
                # 裁剪到边界
                self.data.loc[city_mask, '流量'] = city_data.clip(lower_bound, upper_bound)
                
            elif method == 'remove':
                # 删除异常值
                outlier_mask = (city_data < lower_bound) | (city_data > upper_bound)
                self.data = self.data[~outlier_mask]
                
            elif method == 'replace_with_median':
                # 用中位数替换
                median_val = city_data.median()
                outlier_mask = (city_data < lower_bound) | (city_data > upper_bound)
                self.data.loc[city_mask & outlier_mask, '流量'] = median_val
        
        print("异常值处理完成")
        return self.data
    
    def normalize_data(self, method='minmax', feature_range=(0, 1)):
        """
        数据归一化/标准化
        Args:
            method: 归一化方法 ('minmax', 'standard', 'robust')
            feature_range: 对于minmax的范围
        """
        print(f"\n数据归一化，使用{method}方法...")
        
        self.scalers = {}
        normalized_columns = ['流量']
        
        for column in normalized_columns:
            for city in self.data['地市'].unique():
                city_mask = self.data['地市'] == city
                city_data = self.data.loc[city_mask, column].values.reshape(-1, 1)
                
                if method == 'minmax':
                    scaler = MinMaxScaler(feature_range=feature_range)
                elif method == 'standard':
                    scaler = StandardScaler()
                elif method == 'robust':
                    from sklearn.preprocessing import RobustScaler
                    scaler = RobustScaler()
                
                scaled_data = scaler.fit_transform(city_data)
                self.data.loc[city_mask, f'{column}_normalized'] = scaled_data.flatten()
                self.scalers[f'{city}_{column}'] = scaler
        
        # 更新处理后的数据
        self.processed_data = self.data.copy()
        
        print(f"归一化完成，新增列: 流量_normalized")
        return self.data
    
    def save_processed_data(self, output_path='data/processed_data.csv'):
        """保存处理后的数据"""
        self.processed_data.to_csv(output_path, index=False)
        print(f"\n处理后的数据已保存到: {output_path}")
        return output_path
    
    def visualize_preprocessing(self):
        """可视化预处理效果"""
        if self.data is None or self.data.empty:
            print("没有数据可用于可视化")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 缺失值处理前后对比
        if len(self.data) >= 100:
            time_subset = self.data['时间'].iloc[:100].values
            flow_subset = self.data['流量'].iloc[:100].values
            
            axes[0, 0].plot(time_subset, flow_subset, 'b-', alpha=0.7, label='原始')
            
            if '流量' in self.data.columns:
                ffill_data = self.data['流量'].ffill().iloc[:100].values
                axes[0, 0].plot(time_subset, ffill_data, 'r--', alpha=0.5, label='填充后')
            
            axes[0, 0].set_title('缺失值处理前后对比')
            axes[0, 0].set_xlabel('时间')
            axes[0, 0].set_ylabel('流量')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            plt.setp(axes[0, 0].get_xticklabels(), rotation=45)
        else:
            axes[0, 0].text(0.5, 0.5, '数据不足', ha='center', va='center')
            axes[0, 0].set_title('缺失值处理前后对比')
        
   
        cities = self.data['地市'].unique()
        
        if len(cities) > 0:
            city_data_list = []
            city_labels = []
            
            for i, city in enumerate(cities[:3]):
                city_data = self.data[self.data['地市'] == city]['流量'].dropna()
                if len(city_data) > 0:
                    city_data_list.append(city_data)
                    city_labels.append(city)
            
            if city_data_list:
                positions = list(range(len(city_data_list)))
                axes[0, 1].boxplot(city_data_list, positions=positions)
                axes[0, 1].set_xticks(positions)
                axes[0, 1].set_xticklabels(city_labels)
                axes[0, 1].set_title('各城市流量箱线图')
                axes[0, 1].set_ylabel('流量')
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(0.5, 0.5, '没有有效数据', ha='center', va='center')
                axes[0, 1].set_title('各城市流量箱线图')
        else:
            axes[0, 1].text(0.5, 0.5, '没有城市数据', ha='center', va='center')
            axes[0, 1].set_title('各城市流量箱线图')
        
        # 3. 归一化前后对比
        if '流量_normalized' in self.data.columns:
            original_data = self.data['流量'].dropna()
            normalized_data = self.data['流量_normalized'].dropna()
            
            if len(original_data) > 0 and len(normalized_data) > 0:
                bins = 50
                axes[1, 0].hist(original_data, bins=bins, alpha=0.7, label='原始', density=True)
                axes[1, 0].hist(normalized_data, bins=bins, alpha=0.7, label='归一化', density=True)
                axes[1, 0].set_title('归一化前后分布对比')
                axes[1, 0].set_xlabel('值')
                axes[1, 0].set_ylabel('密度')
                axes[1, 0].legend()
            else:
                axes[1, 0].text(0.5, 0.5, '没有归一化数据', ha='center', va='center')
                axes[1, 0].set_title('归一化前后分布对比')
        else:
            axes[1, 0].text(0.5, 0.5, '未进行归一化', ha='center', va='center')
            axes[1, 0].set_title('归一化前后分布对比')
        
        # 4. 时间序列趋势
        cities = self.data['地市'].unique()
        if len(cities) > 0:
            for city in cities[:3]:
                city_data = self.data[self.data['地市'] == city].copy()
                city_data = city_data.sort_values('时间')
                
                if len(city_data) >= 24*7:
                    if '流量_normalized' in city_data.columns:
                        flow_data = city_data['流量_normalized'].iloc[:24*7].values
                    else:
                        flow_data = city_data['流量'].iloc[:24*7].values
                    
                    time_data = city_data['时间'].iloc[:24*7].values
                    axes[1, 1].plot(time_data, flow_data, label=city, alpha=0.7)
            
            if axes[1, 1].has_data():
                axes[1, 1].set_title('各城市一周流量趋势')
                axes[1, 1].set_xlabel('时间')
                axes[1, 1].set_ylabel('流量')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                plt.setp(axes[1, 1].get_xticklabels(), rotation=45)
            else:
                axes[1, 1].text(0.5, 0.5, '数据不足或没有有效数据', ha='center', va='center')
                axes[1, 1].set_title('各城市一周流量趋势')
        else:
            axes[1, 1].text(0.5, 0.5, '没有城市数据', ha='center', va='center')
            axes[1, 1].set_title('各城市一周流量趋势')
        
        plt.tight_layout()
        try:
            plt.savefig('preprocessing_visualization.png', dpi=300, bbox_inches='tight')
            print("可视化图表已保存到: preprocessing_visualization.png")
        except Exception as e:
            print(f"保存图表时出错: {e}")
        
        plt.show()

# 使用示例
if __name__ == "__main__":
    # 初始化预处理器
    preprocessor = DataPreprocessor(
        data_path='data/data.csv',
        holiday_path='data/holiday_dates.csv'
    )
    
    # 加载数据
    data, holiday_data = preprocessor.load_data()
    
    # 数据探索
    preprocessor.explore_data()
    
    # 处理缺失值
    preprocessor.handle_missing_values(method='interpolate')
    
    # 检测异常值
    outliers_info = preprocessor.detect_outliers(method='iqr')
    
    # 处理异常值
    preprocessor.handle_outliers(method='clip')
    
    # 数据归一化
    preprocessor.normalize_data(method='minmax')
    
    # 保存处理后的数据
    preprocessor.save_processed_data()
    
    # 可视化预处理效果
    preprocessor.visualize_preprocessing()