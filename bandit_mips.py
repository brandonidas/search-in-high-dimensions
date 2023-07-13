import numpy as np

def BanditMIPS(v, query, error_probability, variance_proxy):
    op_count = 0

    d = query.shape[0]
    n = v.shape[0]
    Ssolution = list(range(len(v)-1))
    dused = 0
    confidence_interval_on_current_dimension = np.inf* np.ones(n)
    mu_hat = np.zeros(n)

    while dused < d and len(Ssolution) > 1:
        J = np.random.randint(0, d)  
        op_count += 1
        for i in Ssolution:
          mu_hat[i] = (dused * mu_hat[i] + v[i][J] * query[J]) \
            / (dused + 1) # Update mu_hat_i 
          confidence_interval_on_current_dimension[i] = compute_confidence_interval(
           variance_proxy, i, n, dused, error_probability)
          op_count += 5 + 10
        
        new_Ssolution = []
        max_mu_hat = max(mu_hat)
        for i in Ssolution:
          if mu_hat[i] + confidence_interval_on_current_dimension[i] >= max_mu_hat - confidence_interval_on_current_dimension[i]:
            new_Ssolution.append(i)
          op_count += 3
        Ssolution = new_Ssolution
        
        if len(Ssolution) == 1:
          print("singular condition met")
          print("dused = " + str(dused))
          return Ssolution[0], op_count
        dused += 1
    
    print("len(Ssolution) after successive UCB elimination: " 
          + str(len(Ssolution)))
    print("dused = " + str(dused))

    if len(Ssolution) > 1:
        return exhaustive_inner_product_search(v, query, Ssolution), op_count + len(Ssolution * d)
    else:
        print("Error: None Found")

def compute_confidence_interval(variance_proxy, i, n, dused, error_probability):
   C_dused = variance_proxy[i] * np.sqrt(2 * np.log(4 * n * (max(1, dused) ** 2) / error_probability) / (dused + 1))  # Update C_dused
   return C_dused

def exhaustive_inner_product_search(v, query, Ssolution):
    actual_inner_products = {}
    max_inner_product_sf, max_index_sf = 0, 0
    for i in Ssolution:
        actual_inner_products[i] = np.dot(v[i], query)
        if actual_inner_products[i] > max_inner_product_sf:
            max_inner_product_sf = actual_inner_products[i]
            max_index_sf = i
    return max_index_sf