# -*- coding: utf-8 -*-
import numpy as np
from scipy.sparse import coo_matrix
import time
import pandas as pd
import matplotlib.pyplot as plt



DATA_PATH = 'WikiData.txt'
def ana_data():
    graph = np.loadtxt(DATA_PATH, dtype = np.int16)
    graph = pd.DataFrame(graph, columns=['out_node', 'in_node'])
    #print(graph.head(10))
    page_outnum = graph.groupby(['out_node'])['in_node'].nunique().\
											reset_index().\
											rename(columns={'in_node':'out_node_num'})
    page_outnum = page_outnum.sort_values(['out_node_num'], ascending=False)
    #print(page_outnum['out_node_num'].describe())
    plt.hist(page_outnum[page_outnum.out_node_num<6]['out_node_num'], 10)
    plt.show()
    
    






def save_block():
    graph = np.loadtxt(DATA_PATH, dtype = np.int16)
    print(np.min(graph))
    N = np.max(graph)
    print(N)
    block_num = int(np.floor(N / 4))
    # build block_num blocks
    for i in range(block_num - 1):
        dest = np.arange(4*i + 1, 4*i + 5)
        mask = np.isin(graph[:,1], dest)
        block_graph = graph[mask]
        src_list = np.unique(block_graph[:,0])
        
        block_matrix = np.zeros([src_list.shape[0],6], dtype = np.int16)
        for idx, src in enumerate(src_list):
            block_matrix[idx, 0] = src
            out_node_idx = np.where(graph[:,0] == src)[0]
            out_node = graph[out_node_idx][:,1]      
            out_node_num = out_node.shape[0]
            block_matrix[idx, 1] = out_node_num
            for dest_idx in range(2,6):
                block_dest = 4*i + dest_idx - 1
                if(np.where(out_node == block_dest)[0].shape[0] == 0):    
                    block_matrix[idx, dest_idx] = 0
                else:
                    block_matrix[idx, dest_idx] = 1
            print(block_matrix[idx])
        np.savetxt('cache/block_matrix_'+str(i)+'.txt', block_matrix)
    last_block = block_num - 1
    dest = np.arange(4*last_block + 1, 4*last_block + 6)
    mask = np.isin(graph[:,1], dest)
    block_graph = graph[mask]
    src_list = np.unique(block_graph[:,0])
    
    block_matrix = np.zeros([src_list.shape[0],7], dtype = np.int16)
    for idx, src in enumerate(src_list):
        block_matrix[idx, 0] = src
        out_node_idx = np.where(graph[:,0] == src)[0]
        out_node = graph[out_node_idx][:,1]      
        out_node_num = out_node.shape[0]
        block_matrix[idx, 1] = out_node_num
        for dest_idx in range(2,7):
            block_dest = 4*last_block + dest_idx - 1
            if(np.where(out_node == block_dest)[0].shape[0] == 0):
                block_matrix[idx, dest_idx] = 0
            else:
                block_matrix[idx, dest_idx] = 1
        print(block_matrix[idx])
    np.savetxt('cache/block_matrix_'+str(last_block)+'.txt', block_matrix)
def PageRank():
    beta = 0.85
    N = 8297
    debug = 1
    block_num = int(np.floor(N / 4))
    
    for iter in range(20000):
        S = 0
        total_error = 0
        if(iter == 0):
            for i in range(block_num - 1):
                np.savetxt('cache/r_old_' + str(i) + '.txt', np.full([4,1], 1 / float(N), dtype = np.float32))
            np.savetxt('cache/r_old_' + str(block_num - 1) + '.txt', np.full([5,1], 1 / float(N), dtype = np.float32))    
        
        for i in range(block_num - 1):
            dest_list = np.arange(4*i, 4*i + 4)
            block_matrix = np.loadtxt('cache/block_matrix_'+str(i)+'.txt')
            if(block_matrix.ndim == 1 and block_matrix.shape[0] > 0):            
                block_matrix = block_matrix.reshape([1,6])         
            r_new = np.full([4,1], 0, dtype = np.float32)
            r_old = np.loadtxt('cache/r_old_' + str(i) + '.txt')
            for src_idx in range(block_matrix.shape[0]):                        
                src = int(block_matrix[src_idx][0])
                out_node_num = block_matrix[src_idx][1]
                block_num_src = min((src - 1) / 4, 2073)
                r_old_temp = np.loadtxt('cache/r_old_' + str(block_num_src) + '.txt')
                block_src_idx = src - block_num_src*4 - 1
                dst_idx = np.where(block_matrix[src_idx][2:6] == 1)                
                
                
                r_new[dst_idx] = r_new[dst_idx] + beta * r_old_temp[block_src_idx] / out_node_num 
            S = S + np.sum(r_new)        
            np.savetxt('cache/r_new_' + str(i) + '.txt', r_new)            
        last_block = block_num - 1
        dest_list = np.arange(4*last_block, 4*last_block + 5)
        block_matrix = np.loadtxt('cache/block_matrix_'+str(last_block)+'.txt')
        r_new = np.full([5,1], 0, dtype=np.float32)
        r_old = np.loadtxt('cache/r_old_' + str(last_block) + '.txt')
        
        for src_idx in range(block_matrix.shape[0]):
            src = int(block_matrix[src_idx][0])
            out_node_num = block_matrix[src_idx][1]
            block_num_src = min((src - 1) / 4, 2073)
            r_old_temp = np.loadtxt('cache/r_old_' + str(block_num_src) + '.txt')
            block_src_idx = src - block_num_src*4 - 1
            dst_idx = np.where(block_matrix[src_idx][2:7] == 1) 
            r_new[dst_idx] = r_new[dst_idx] + beta * r_old_temp[block_src_idx] / out_node_num
        S = S + np.sum(r_new)
        np.savetxt('cache/r_new_' + str(last_block) + '.txt', r_new)
        
        for i in range(block_num):
            r_new = np.loadtxt('cache/r_new_' + str(i) + '.txt')            
            r_new = r_new + (1 - S) / N * np.ones_like(r_new)
            r_old = np.loadtxt('cache/r_old_' + str(i) + '.txt')
            total_error = total_error + np.sum(np.abs(r_old - r_new))
            np.savetxt('cache/r_old_' + str(i) + '.txt', r_new)
        r_old = np.loadtxt('cache/r_old_0.txt')
        for i in np.arange(1, block_num ):
            r_old = np.concatenate([r_old, np.loadtxt('cache/r_old_' + str(i) + '.txt')])
        print(total_error)
        if (total_error < 1e-8):
            break
    
       

def makeRes():
    N = 8297
    block_num = int(np.floor(N / 4))
    r_old = np.loadtxt('cache/r_old_0.txt')
    for i in np.arange(1, block_num ):
        r_old = np.concatenate([r_old, np.loadtxt('cache/r_old_' + str(i) + '.txt')])
        
    print(r_old.shape)
    result = np.sort(-r_old)
    result = -result[:100]
    result_idx = np.argsort(-r_old)
    result_idx = result_idx[:100] + np.ones_like(result_idx[:100]) 
    result = result.reshape([100, 1])
    result_idx = result_idx.reshape([100, 1])
    final_result = np.concatenate([result_idx, result], axis = 1)
    with open('result.txt','w') as f:
        f.write('NodeID'+'\t'+'Score'+'\r\n')
        for i in range(100):
            f.write(str(int(final_result[i][0]))+'\t'+str(final_result[i][1])+'\r\n')
    

    
        
        

    
    


if __name__ == "__main__":
    #save_block()
    #PageRank()
    makeRes()
    #ana_data()
    









