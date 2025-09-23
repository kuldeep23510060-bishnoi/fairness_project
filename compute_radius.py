

def compute_radius(alpha, train_df, protected_col):

    tv_if = 2.0 * alpha - alpha * alpha
    l1_radius_if = 2.0 * tv_if
    
    pi = train_df[protected_col].value_counts(normalize=True).to_dict()

    gamma_js = {j: (alpha / ((1 - alpha) * pj)) if pj > 0 else 0.0 for j, pj in pi.items()}
    max_dp_radius = 2*[gamma_js.get(0, 0.0), gamma_js.get(1, 0.0)]

    tv0_dp = max_dp_radius[0]
    tv1_dp = max_dp_radius[1]

    l1_radius_dp = max(tv0_dp, tv1_dp)
    
    return l1_radius_dp, l1_radius_if

