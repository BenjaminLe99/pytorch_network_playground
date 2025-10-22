
def expand(str_list):
    def brace_expand(s):
        # "a{b,c}d" -> ["abd", "acd"]
        if "{" not in s:
            return [s]
        pre, post = s.split("{", 1)
        mid, post = post.split("}", 1)
        parts = mid.split(",")
        expanded = []
        for part in parts:
            for rest in brace_expand(post):
                expanded.append(pre + part + rest)
        return expanded
    cols = []
    for _str in str_list:
        cols.extend(brace_expand(_str))
    return cols


data_prefix = "res_dnn_pnet_"

categorical_features: list[str] = [
    "pair_type",
    "dm1",
    "dm2",
    "vis_tau1_charge",
    "vis_tau2_charge",
    "has_jet_pair",
    "has_fatjet",
]

continous_features: list[str] = [
    "met_px", "met_py",
    "met_cov00", "met_cov01", "met_cov11",
    "vis_tau1_px", "vis_tau1_py", "vis_tau1_pz", "vis_tau1_e",
    "vis_tau2_px", "vis_tau2_py", "vis_tau2_pz", "vis_tau2_e",
    "bjet1_px", "bjet1_py", "bjet1_pz", "bjet1_e",
    "bjet1_tag_b", "bjet1_tag_cvsb", "bjet1_tag_cvsl", "bjet1_hhbtag",
    "bjet2_px", "bjet2_py", "bjet2_pz", "bjet2_e",
    "bjet2_tag_b", "bjet2_tag_cvsb", "bjet2_tag_cvsl", "bjet2_hhbtag",
    "fatjet_px", "fatjet_py", "fatjet_pz", "fatjet_e",
    "htt_e", "htt_px", "htt_py", "htt_pz",
    "hbb_e", "hbb_px", "hbb_py", "hbb_pz",
    "htthbb_e", "htthbb_px", "htthbb_py", "htthbb_pz",
    "httfatjet_e", "httfatjet_px", "httfatjet_py", "httfatjet_pz",
]

continous_features =  [data_prefix + f for f in continous_features]
categorical_features =  [data_prefix + f for f in categorical_features]
