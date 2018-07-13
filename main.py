# -*- coding: utf-8 -*-
import numpy as np
from scipy.sparse import coo_matrix
import time
import pandas as pd



DATA_PATH = 'WikiData.txt'

def read_raw_data():
    # 从数据文件中读取link信息构建转移矩阵
    data = pd.read_table(DATA_PATH,header=None)
    row = np.array(data[1]) - 1
    col = np.array(data[0]) - 1
    N = np.max(np.max(data))
    values = np.full([len(row)],1)
    print(N) #网络中的节点总数
    #转化为稀疏矩阵
    trans_matrix = coo_matrix((values,(row,col)),shape=(N,N))
    #列向量归一化
    out_link = np.sum(trans_matrix,axis=1) #列向量之和为每个节点的出度
    #转换为01序列
    mask = np.reshape(np.ceil(np.true_divide(out_link,out_link+1)),(1,N))
    mask = np.reshape(1 - mask,(1,N))
    # 让全为0的列变为全为1
    trans_matrix = trans_matrix + mask
    #重新计算每个节点的出度 让原来出度为0的节点，指向网络中所有节点，解决dead end
    out_link = np.reshape(np.sum(trans_matrix,axis=1),(1,N))
    trans_matrix = np.true_divide(trans_matrix,out_link)
    return trans_matrix,N


def pageRank(trans_matrix,N):
    error = 100 #累计误差
    epoch = 1 #迭代次数
    beta = 0.85
    lbeta = 1 - beta
    pagerank_vec = np.full([N,1],1/float(N),dtype=np.float32)
    #随机游走向量，避免spider traps 初始化为 0.15 * 1/N
    en = np.full([N,1],lbeta/float(N),dtype=np.float32)
    while(error>1e-12):
        temp = np.multiply(beta,np.matmul(trans_matrix,pagerank_vec)) + en
        error = np.sum(np.abs(pagerank_vec - temp))
        pagerank_vec = temp
        print("epoch:",epoch,"error:",error)
        epoch = epoch + 1
    id = np.arange(1,N+1).astype(int)
    score = list(np.reshape(np.asarray(pagerank_vec),(N)))
    result = pd.DataFrame({"NodeID":list(id),"Score":score})
    result = pd.DataFrame(result.sort_values('Score',ascending=False).head(100))
    print(result)
    with open('result.txt','w') as f:
        f.write('NodeID'+'\t'+'Score'+'\r\n')
        for index,row in result.iterrows():
            f.write(str(int(row[0]))+'\t'+str(row[1])+'\r\n')


if __name__ == "__main__":
    s = time.time()
    matrix,N = read_raw_data()
    pageRank(matrix,N)
    t = time.time()
    print('Cost:',t-s,'s')









