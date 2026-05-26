from SF_TRON_FN.SRC.Config.Config import *

PPOCfg.CriticParam.state_dim = 33 + PPOCfg.EstimatorParam.output_dim
