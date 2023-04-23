import numpy as np

def pairwise2overall(d, m):
    overall_diversity = 2*d/(m*(m-1))
   
    return overall_diversity
def q_statistics(preds, y, bags):
    n = preds.shape[0]
    m = preds.shape[1]
    
    d = 0
    for i, _ in bags.items():
        for j in range(i + 1, m):
            clf1_preds = preds[:, i]
            clf2_preds = preds[:, j]
            n11, n10, n01, n00 = 0, 0, 0, 0
            for k in range(n):
                if clf1_preds[k] == y[k] and clf2_preds[k] == y[k]:
                    n11 += 1
                elif clf1_preds[k] == y[k] and clf2_preds[k] != y[k]:
                    n10 += 1
                elif clf1_preds[k] != y[k] and clf2_preds[k] == y[k]:
                    n01 += 1
                elif clf1_preds[k] != y[k] and clf2_preds[k] != y[k]:
                    n00 += 1
            numerator = n11*n00 - n10*n01
            if numerator > 0:
                d += numerator/(n11*n00 + n10*n01)
                
            else:
                d += 1
        overall_diversity = pairwise2overall(d, m)   
        bags[i]['diverse']=overall_diversity
        
    
    