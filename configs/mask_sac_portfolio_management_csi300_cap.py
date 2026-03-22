# base parameters (do not modify)
import time
import datetime

root = None
workdir = "workdir"
tag = "mask_sac"
num_stocks = 300
num_envs = 1
test_num_envs = 4
num_features = 154 # num features name + num temporals name
temporal_dim = 3 # weekday, day, month
train_start_date = "2016-09-01"
train_end_date = "2019-12-10"
val_start_date = "2019-12-11"
val_end_date = "2021-05-19"
test_start_date = "2021-05-20"
test_end_date = "2025-07-01"
offset = 64
train_size = 1408
valid_size = 733 #815
test_size = 0
# train_start_date = "2017-01-01"
# train_end_date = "2021-12-10"
# val_start_date = "2021-12-11"
# val_end_date = "2022-08-05"
# test_start_date = "2022-08-06"
# test_end_date = "2025-03-01"
# offset = 64
# train_size = 1408
# valid_size = 643
# test_size = 0
# train_start_date = "2007-09-26"
# val_start_date = "2019-07-22"
# test_start_date = "2021-01-08"
# test_end_date = None
if_use_per = False
if_use_rep = True
if_use_beta = True
if_norm = True
if_norm_temporal = False
save_freq = 20
repeat_times = 2
action_wrapper_method = "softmax"
T = 1
n_steps_per_episode = 1024
rolling_split_path = '/root/quant-ml-qlib/Graph-EarnMore/datasets/csi300/features/SH600011.csv'
#rolling_split_path = '/root/quant-ml-qlib/EarnMore/datasets/ndx/features/AAPL.csv'

# train parameters (adjust mainly)
num_episodes = 300
days = 1
batch_size = 128
buffer_size = 4000
horizon_len = 128
embed_dim = 128
decoder_embed_dim = 512
depth = 2 # 1 transformer
decoder_depth = 2
lr = 5e-4 # act_lr, cri_lr
act_lr = 5e-4
cri_lr = 5e-4
rep_lr = 5e-4
beta_lr = 5e-4
rep_loss_weight = 0.01
beta_loss_weight = 0.01
seed = 10

# size
feature_size = (days, num_features)
patch_size = (days, num_features)

transition = ["state", "rep_state","action", "node", "reward", "done", "next_rep_state"]
# transition_shape = dict(
#     state = dict(shape = (num_envs, num_stocks, days, num_features), type = "float32"),
#     action = dict(shape = (num_envs, num_stocks+1), type = "float32"),
#     mask = dict(shape = (num_envs, num_stocks), type = "int32"),
#     ids_restore = dict(shape = (num_envs, num_stocks), type = "int64"),
#     reward = dict(shape = (num_envs, ), type = "float32"),
#     done = dict(shape = (num_envs, ), type = "float32"),
#     next_state  = dict(shape = (num_envs, num_stocks, days, num_features), type = "float32")
# )

transition_shape = dict(
    state = dict(shape = (num_envs, num_stocks, days, num_features), type = "float32"),
    rep_state = dict(shape = (num_envs, num_stocks, 512), type = "float32"),
    action = dict(shape = (num_envs, num_stocks+1), type = "float32"),
    node = dict(shape = (num_envs, num_stocks), type = "int32"),
    reward = dict(shape = (num_envs, ), type = "float32"),
    done = dict(shape = (num_envs, ), type = "float32"),
    next_rep_state = dict(shape = (num_envs, num_stocks, 512), type = "float32"),
)

current_date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
# current_date_1 = '2025-03-31-20-20-53'
# current_date_2 = '2025-04-27-15-52-04'
rand_seed = int(time.mktime(datetime.datetime.strptime(current_date, '%Y-%m-%d-%H-%M-%S').timetuple()))
dataset = dict(
    type = "PortfolioManagementDataset",
    root = root,
    rand_seed = rand_seed,
    current_date = current_date,
    # data_path= "datasets/dj30/features",
    # stocks_path = "datasets/dj30/stocks.txt",
    # aux_stocks_path = "datasets/dj30/aux_stocks_files",
    data_path="datasets/csi300/features",
    # data_path="datasets/test/features",
    stocks_path="datasets/csi300/stocks.txt",
    aux_stocks_path="datasets/csi300/aux_stocks_files",
    temporals_name = [
        "weekday",
        "day",
        "month",
    ],
    labels_name = [
        'ret1',
        'mov1',
    ]
)

environment = dict(
    type = "EnvironmentPV",
    dataset = None,
    mode = "train",
    if_norm = if_norm,
    if_norm_temporal = if_norm_temporal,
    scaler = None,
    days = days,
    start_date = None,
    end_date = None,
    initial_amount = 1e5,
    transaction_cost_pct = 1e-3,#
)

# rep_net = dict(
#     type = "MaskTimeState",
#     embed_type = "TimesEmbed",
#     feature_size = feature_size,
#     patch_size = patch_size,
#     t_patch_size = 1,
#     num_stocks = num_stocks,
#     pred_num_stocks = num_stocks,
#     in_chans = 1,
#     input_dim=num_features,
#     temporal_dim=temporal_dim,
#     embed_dim = embed_dim,
#     depth = depth,
#     num_heads = 4,
#     decoder_embed_dim = decoder_embed_dim,
#     decoder_depth = decoder_depth,
#     decoder_num_heads = 8,
#     mlp_ratio = 4.0,
#     norm_pix_loss = False,
#     cls_embed = True,
#     sep_pos_embed = True,
#     trunc_init = False,
#     no_qkv_bias = False,
#     mask_ratio_min = 0.05,
#     mask_ratio_max = 1,
#     mask_ratio_mu = 0.2,#0.6
#     mask_ratio_std = 0.1,#0.5
# )

rep_net = dict(
        type = "GraphMaskTimeState",
        in_dim = num_features,
        nhead_out = 1,
        num_hidden= 512,
        nhead= 2,
        mask_rate= 0.4,
        num_layers= 4,
        encoder_type= 'gat',
        decoder_type= 'gat',
        activation= 'prelu',
        negative_slope = 0.2,
        loss_fn= 'sce',
        replace_rate= 0.0,
        drop_edge_rate= 0.0,
        alpha_l= 3,
        attn_drop=0.01,
        norm= 'layernorm',
        concat_hidden = False,
        feat_drop = 0.01,
        residual= True,
        mask_ratio_min = 0.05,
        mask_ratio_max = 0.9,
        mask_ratio_mu = 0.3,#0.6
        mask_ratio_std = 0.2,#0.5
    )

act_net = dict(
        type = "ActorMaskSAC",
        embed_dim = decoder_embed_dim,
        depth = depth,
        cls_embed = True,
    )

cri_net = dict(
        type = "CriticMaskSAC",
        embed_dim = decoder_embed_dim,
        depth = depth,
        cls_embed = True,
    )

criterion = dict(type='MSELoss', reduction="none")
scheduler = dict(type='MultiStepLRScheduler',
                 multi_steps=[600 * n_steps_per_episode, 1000 * n_steps_per_episode, 1400 * n_steps_per_episode],
                 t_initial = num_episodes * n_steps_per_episode,
                 decay_t = 500 * n_steps_per_episode,
                 gamma = 0.1,
                 t_mul = 1.,
                 lr_min = 0.,
                 decay_rate = 1.,
                 warmup_t = 300 * n_steps_per_episode,
                 warmup_lr_init = 1e-8,
                 warmup_prefix = False,
                 cycle_limit = 0,
                 t_in_epochs = False,
                 noise_range_t = None,
                 noise_pct = 0.67,
                 noise_std = 1.0,
                 noise_seed = rand_seed,
                 initialize = True)
optimizer = dict(type='AdamW',
                 params = None,
                 lr=lr)

agent = dict(
    type = "AgentMaskSAC",
    act_lr=act_lr,
    cri_lr=cri_lr,
    rep_lr=rep_lr,
    beta_lr=beta_lr,
    rep_net = rep_net,
    act_net = act_net,
    cri_net = cri_net,
    criterion = criterion,
    optimizer = optimizer,
    scheduler = scheduler,
    if_use_per=if_use_per,
    if_use_rep=if_use_rep,
    if_use_beta=if_use_beta,
    rep_loss_weight=rep_loss_weight,
    beta_loss_weight=beta_loss_weight,
    num_envs = num_envs,
    transition_shape = transition_shape,
    max_step = 1e4,
    gamma = 0.99,
    reward_scale = 0.99,
    repeat_times = repeat_times,
    batch_size = batch_size,
    clip_grad_norm = 3.0,
    soft_update_tau = 5e-3,
    state_value_tau = 0,
    device = None,
    action_wrapper_method = action_wrapper_method,
    T = T,
)
