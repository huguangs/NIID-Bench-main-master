import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
# 读取基因表达谱数据
data = pd.read_csv('BRCA_expression.csv')

# 提取基因表达数据和临床数据
gene_expr = data.iloc[:, 1:].T
clinical_data = data[['patient_id', 'vital_status', 'days_to_death', 'days_to_last_follow_up']]

# 数据预处理
gene_expr = gene_expr.apply(lambda x: x / x.sum(), axis=1)  # 归一化
clinical_data['vital_status'] = clinical_data['vital_status'].apply(lambda x: 1 if x=='dead' else 0)  # 生存状态编码
clinical_data[['days_to_death', 'days_to_last_follow_up']] = clinical_data[['days_to_death', 'days_to_last_follow_up']].fillna(0)  # 缺失值填充


# 进行PCA分析
pca = PCA(n_components=2)
pca_data = pca.fit_transform(gene_expr)
plt.scatter(pca_data[:, 0], pca_data[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA plot')
plt.show()

# 进行聚类分析
kmeans = KMeans(n_clusters=2, random_state=0)
labels = kmeans.fit_predict(gene_expr)
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Cluster plot')
plt.show()



# 计算生存时间和生存状态
time = clinical_data['days_to_death'] + clinical_data['days_to_last_follow_up']
event = clinical_data['vital_status']

# 进行Kaplan-Meier生存分析
kmf = KaplanMeierFitter()
kmf.fit(time, event)
kmf.plot()
plt.xlabel('Time (days)')
plt.ylabel('Survival probability')
plt.title('Kaplan-Meier survival curve')
plt.show()

# 进行Cox比例风险模型分析
cox = CoxPHFitter()
cox.fit(clinical_data[['vital_status', 'days_to_death', 'days_to_last_follow_up']], 'days_to_death')
cox.print_summary()
