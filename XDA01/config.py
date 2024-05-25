import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="./.env",verbose=True)

class Settings:
    # Service API Info
    SERVICE_URL = os.getenv("SERVICE_URL")
    SERVICE_PORT = os.getenv("SERVICE_PORT")

    # DataBase Info
    DB_USERNAME : str = os.getenv("DB_USERNAME")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST : str = os.getenv("DB_HOST")
    DB_PORT : str = os.getenv("DB_PORT")
    DB_DATABASE : str = os.getenv("DB_DATABASE")
    DATABASE_URL = os.getenv("DATABASE_URL")

class Modelargs:
    # basic config
    is_training      = 0                # 'status'
    train_only       = False,          # perform training on full input dataset without validation and testing
    model_id         = 'test'           # model id
    model            = 'DLinear'     # model name, options: [Autoformer, Informer, Transformer]
    # data loader
    data             = 'ETTh1'           # dataset type
    root_path        = './DB/storage/ETT/'     # root path of the data file
    data_path        = 'ETTh1.csv'       # data file
    direct_pred_input = None
    features         = 'M'               # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
    target           = 'OT'              # target feature in S or MS task
    freq             = 'h'               # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
    checkpoints      = './MLmodels/checkpoints/'  # location of model checkpoints
    # forecasting task
    seq_len          = 336        # input sequence length
    label_len        = 96        # start token length
    pred_len         = 96        # prediction sequence length
    # DLinear
    individual       = False     # DLinear: a linear layer for each variate(channel) individually
    # Formers 
    embed_type       = 0         # 0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding
    enc_in           = 7         # encoder input size  # DLinear with --individual, use this hyperparameter as the number of channels
    dec_in           = 7         # decoder input size
    c_out            = 7         # output size
    d_model          = 512       # dimension of model
    n_heads          = 8         # num of heads
    e_layers         = 2         # num of encoder layers
    d_layers         = 1         # num of decoder layers
    d_ff             = 2048      # dimension of fcn
    moving_avg       = 25        # window size of moving average
    factor           = 1         # attn factor
    distil           = False     # whether to use distilling in encoder, using this argument means not using distilling
    dropout          = 0.05      # dropout
    embed            = 'timeF'   # time features encoding, options:[timeF, fixed, learned]'
    activation       = 'gelu'    # activation
    output_attention = True      # whether to output attention in ecoder
    do_predict       = False      # whether to predict unseen future data
    # optimization
    num_workers   = 1        # data loader num workers
    itr           = 1         # experiments times
    train_epochs  = 100        # train epochs
    batch_size    = 32        # batch size of train input data
    patience      = 7         # early stopping patience
    learning_rate = 0.005   # optimizer learning rate
    des           = 'Exp'    # exp description
    loss          = 'mse'     # loss function
    lradj         = 'type1'   # adjust learning rate
    use_amp       = False     # use automatic mixed precision training
    # GPU
    use_gpu       = True      # use gpu
    gpu = 0
    use_multi_gpu = False
    devices       = '0,1,2,3' # device ids of multile gpus
    test_flop     = False     # See utils/tools for usage

