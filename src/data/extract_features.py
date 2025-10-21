
translation = {
    "pair_type" : "res_dnn_pnet_pair_type",
    "decay_mode1" : "res_dnn_pnet_dm1",
    "decay_mode2" : "res_dnn_pnet_dm2",
    "lepton1.charge" : "res_dnn_pnet_vis_tau1_charge",
    "lepton2.charge" : "res_dnn_pnet_vis_tau2_charge",
    "has_fatjet" : "res_dnn_pnet_has_fatjet",
    "has_jet_pair" : "res_dnn_pnet_has_jet_pair",
    "year_flag" : "missing",
    "bjet1.btagPNetB" : "res_dnn_pnet_bjet1_tag_b",
    "bjet1.btagPNetCvB" : "res_dnn_pnet_bjet1_tag_cvsb",
    "bjet1.btagPNetCvL" : "res_dnn_pnet_bjet1_tag_cvsl",
    "bjet1.energy" : "res_dnn_pnet_bjet1_energy",
    "bjet1.hhbtag" : "res_dnn_pnet_bjet1_hhbtag",
    "bjet1.mass" : "res_dnn_pnet_bjet1_mass",
    "bjet1.px" : "res_dnn_pnet_bjet1_px",
    "bjet1.py" : "res_dnn_pnet_bjet1_py",
    "bjet1.pz" : "res_dnn_pnet_bjet1_pz",
    "bjet2.btagPNetB" :  "res_dnn_pnet_bjet2_tag_b",
    "bjet2.btagPNetCvB" : "res_dnn_pnet_bjet2_tag_cvsb",
    "bjet2.btagPNetCvL" : "res_dnn_pnet_bjet2_tag_cvsl",
    "bjet2.energy" : "res_dnn_pnet_bjet2_energy",
    "bjet2.hhbtag" : "res_dnn_pnet_bjet2_hhbtag",
    "bjet2.mass" : "res_dnn_pnet_bjet2_mass",
    "bjet2.px" : "res_dnn_pnet_bjet2_px",
    "bjet2.py" : "res_dnn_pnet_bjet2_py",
    "bjet2.pz" : "res_dnn_pnet_bjet2_pz",
    "fatjet.energy" : "res_dnn_pnet_fatjet_e",
    "fatjet.mass" : "missing",
    "fatjet.px" : "res_dnn_pnet_fatjet_px",
    "fatjet.py" : "res_dnn_pnet_fatjet_py",
    "fatjet.pz" : "res_dnn_pnet_fatjet_pz",
    "lepton1.energy" : "res_dnn_pnet_vis_tau1_e",
    "lepton1.mass" : "res_dnn_pnet_vis_tau1_mass",
    "lepton1.px" : "res_dnn_pnet_vis_tau1_px",
    "lepton1.py" : "res_dnn_pnet_vis_tau1_py",
    "lepton1.pz" : "res_dnn_pnet_vis_tau1_pz",
    "lepton2.energy" : "res_dnn_pnet_vis_tau2_e",
    "lepton2.mass" : "res_dnn_pnet_vis_tau2_mass",
    "lepton2.px" : "res_dnn_pnet_vis_tau2_px",
    "lepton2.py" : "res_dnn_pnet_vis_tau2_py",
    "lepton2.pz" : "res_dnn_pnet_vis_tau2_pz",
    "PuppiMET.px" : "res_dnn_pnet_met_px",
    "PuppiMET.py" : "res_dnn_pnet_met_py",
    "reg_dnn_nu1_px" : "reg_dnn_nu1_px",
    "reg_dnn_nu1_py" : "reg_dnn_nu1_py", # missing
    "reg_dnn_nu1_pz" : "reg_dnn_nu1_pz",
    "reg_dnn_nu2_px" : "reg_dnn_nu2_px",
    "reg_dnn_nu2_py" : "reg_dnn_nu2_py", # missing
    "reg_dnn_nu2_pz" : "reg_dnn_nu2_pz",
    }


def translate_names(columns):
    return [translation[col] for col in columns]

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


# continous_features = expand([
#     "bjet1.{btagPNetB,btagPNetCvB,btagPNetCvL,hhbtag,px,py,pz}",
#     "bjet2.{btagPNetB,btagPNetCvB,btagPNetCvL,hhbtag,px,py,pz}",
#     "fatjet.{px,py,pz}",
#     "lepton1.{px,py,pz}",
#     "lepton2.{px,py,pz}",
#     "PuppiMET.{px,py}",
#     # "reg_dnn_nu{1,2}_{px,pz}",
# ])

# categorical_features = expand([
# "pair_type",
# "decay_mode{1,2}",
# "lepton{1,2}.charge",
# "has_fatjet",
# "has_jet_pair",
# # "year_flag ",
# ])

categorical_features: list[str] = [
    "pair_type",
    "dm1",
    "dm2",
    "vis_tau1_charge",
    "vis_tau2_charge",
    "has_jet_pair",
    "has_fatjet",
]
# continuous input features to the network
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


# continous_features = expand([
#     "bjet1.{btagPNetB,btagPNetCvB,btagPNetCvL,energy,hhbtag,px,py,pz}",
#     "bjet2.{btagPNetB,btagPNetCvB,btagPNetCvL,energy,hhbtag,px,py,pz}",
#     "fatjet.{energy,px,py,pz}",
#     "lepton1.{energy,px,py,pz}",
#     "lepton2.{energy,px,py,pz}",
#     "PuppiMET.{px,py}",
#     "reg_dnn_nu{1,2}_{px,pz}",
# ])

# categorical_features = expand([
# "pair_type",
# "decay_mode{1,2}",
# "lepton{1,2}.charge",
# "has_fatjet",
# "has_jet_pair",
# "year_flag ",
# ])
