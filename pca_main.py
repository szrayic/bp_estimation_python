# -*-coding: utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import pca_vector_number
# 导入matlab文件
matfn=u'/home/wangbo/特征数据/realdata3.mat'
rawdata=sio.loadmat(matfn)  # 得到的是字典类型
# 导出所有特征
ppgfeatures=rawdata['PPG_Features']  #这是一个 long×31 的
ecgfeatures=rawdata['ECG_Features']  #这是一个 long×5 的
alingfeatures=rawdata['PPG_ECG_alingFeatures']  #这是一个 long×3 的
sys_bp=rawdata['SYS_BP'].transpose()  # 这里的数据的二维ndarray数据 1*long,再转置
dia_bp=rawdata['DIA_BP'].transpose()
# 变ndarray为DataFrame
a=pd.DataFrame(ppgfeatures,columns=['PH','DNH','DNHr','DPH','DPHr','RBW','RBW10','RBW25','RBW33','RBW50','RBW75','DBW10','DBW25','DBW33','DBW50','DBW75','PDNT','DNDPT','KVAL','PWA','RBAr','DBAr','DiaAr','SLP1','SLP2','SLP3','AmBE','E','KTEMIU','KTEDELTA','ENTROP'])
b=pd.DataFrame(ecgfeatures,columns=['T_QR','T_RS','T_RR','H_QR','H_RS'])
c=pd.DataFrame(alingfeatures,columns=['PWTT_RP','PWTT_RO','PWTT_RhalfP'])
d=pd.DataFrame(sys_bp,columns=['SYS_BP'])
e=pd.DataFrame(dia_bp,columns=['DIA_BP'])
data=pd.concat([a,b,c,d,e],axis=1)
# 将数据划分为训练数据集和测试集
x=data.iloc[:,0:39]
y=data.iloc[:,[39,40]]
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.1,random_state=0) #70%训练

# 特征数据标准化
stdsc = StandardScaler()
x_train_std = stdsc.fit_transform(x_train)
x_test_std = stdsc.transform(x_test)

# 构造协方差矩阵
cov_mat = np.cov(x_train_std,rowvar=0)
eigen_vals, engen_vecs = np.linalg.eig(cov_mat)
print eigen_vals
number=pca_vector_number.percentage2n(eigen_vals,0.95)
print number
# 要几个特征向量
tot=sum(eigen_vals)
var_exp=[(i/tot) for i in sorted(eigen_vals,reverse=True)]
cum_var_exp=np.cumsum(var_exp)
plt.bar(range(1,40),var_exp,alpha=0.5,align='center')
plt.step(range(1,40),cum_var_exp,where='mid')
plt.show()