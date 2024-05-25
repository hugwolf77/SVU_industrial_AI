# application main.py

import sys

import torch
from MLmodels.DLM import DLM

from api.service import apiRun
from config import Settings, Modelargs

from pytz import timezone
from datetime import datetime
today = datetime.now(timezone('Asia/Seoul'))

# from DB import migrate as mig
# from DB.DBmodel.dataTB import Base, ETT_H_1, ETT_H_2, ETT_M_1, ETT_M_2 

configs = Settings()

service_url = configs.SERVICE_URL
service_port = configs.SERVICE_PORT


# def init_DB_and_TB():
#     # init DataBase
#     mig.init_DB()
#     # raw data file load to DataFrame
#     dataTags = {'ETT':'/DB/storage/ETDataset-raw/'}
#     dfList = mig.rawData(dataTags)

#     # migration rawData to DB Table
#     nameTB =  ['ETT_H_1', 'ETT_H_2', 'ETT_M_1', 'ETT_M_2' ]
#     for df in zip(nameTB,dfList):
#         mig.migrate(df[0], df[1])

#     # comfirm migrated Table
#     migTB = [ETT_H_1, ETT_H_2, ETT_M_1, ETT_M_2 ]
#     iDate = '2016-07-01 01:00:00'
#     mig.comfirmTB(migTB[0],iDate)

def model_train_test(trainset):
    args =  Modelargs()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    args.is_training = trainset
    args.itr = 1
    args.do_predict = 0

    print(f"model_args : \n {args}")
    print(args.model_id)

    Exp = DLM

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_pl{}_eb{}_{}'.format(args.model_id, args.model, args.data, args.features, args.seq_len,args.pred_len,args.embed,args.des)
            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            if not args.train_only:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)
            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_pl{}_eb{}_{}'.format(args.model_id, args.model, args.data, args.features, args.seq_len,args.pred_len,args.embed,args.des)
        exp = Exp(args)  # set experiments
        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)
        else:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
        torch.cuda.empty_cache()

if __name__ == "__main__":

    # api start
    if len(sys.argv) == 1:
        apiRun(service_url,service_port)

    elif len(sys.argv) == 2:
        # init_DB_and_TB()
        # if sys.argv[1] == 'DBinit':
        #     print("DataBase initiation......")
        #     init_DB_and_TB()

        # model train
        if sys.argv[1] == 'model_train':
            try:
                if sys.argv[1] == 'model_train':
                    print("model train start......")
                    trainset = 1
                    model_train_test(trainset)
            except Exception as e: 
                print(f"model train error.....\n{e}")
        else:
            print(f"if ['DBinit', 'model_train'] is not selected One option , there are no other options")
    
