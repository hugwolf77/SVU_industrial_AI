# client app by pyside6
import sys
import os
from datetime import datetime
from pytz import timezone

# pyside6 window
from PySide6 import QtCore
from PySide6.QtCore import QFile, QIODevice 
from PySide6.QtWidgets import (QApplication, QWidget, QFileDialog, QTableWidget, QTableWidgetItem)
from PySide6.QtUiTools import QUiLoader

# data read
# from sqlalchemy import select
# from DB.connect import conn_DB
# from DB.DBmodel.dataTB import (Base, ETT_H_1, ETT_H_2, ETT_M_1, ETT_M_2)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

import numpy as np
import pandas as pd
from pydantic import BaseModel
# predict request 
import requests


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, heigth=4, dpi=150):
        fig = Figure(figsize=(width, heigth), dpi=dpi, facecolor='gray')
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class DataInput(BaseModel):
    # ReqTime : str = Field(min_length=4, max_length=10)
    Index : list[int]
    date : list[datetime]
    HUFL : list[float]
    HULL : list[float]
    MUFL : list[float]
    MULL : list[float]
    LUFL : list[float]
    LULL : list[float]
    OT : list[float] 

class Client(QWidget):
    __MAX_WIN = 1
    __INST_created = 0
    
    def __new__(cls):
        if (cls.__INST_created > cls.__MAX_WIN):
            raise ValueError("Cannot create more objects")
        cls.__INST_created += 1
        return super().__new__(cls)
    
    def __init__(self):
        super(Client, self).__init__()
        self.window = self.SetupUI()
        self.file_navi = QFileDialog()
        self.window.setWindowTitle('TSF_Model_Client')
        # select input data file
        self.window.path_edit.returnPressed.connect(self.Input_path)
        self.window.Path_btn.clicked.connect(self.file_exeplore)
        self.window.select.clicked.connect(self.Select_data)
        self.select_data = ''
        # predict 
        self.Input_data = pd.DataFrame()
        self.window.predict.clicked.connect(self.Predict_result)
        self.predict_response = pd.DataFrame()
        ## load data DB connect
        # self.engine = conn_DB()
        
        # Graph
        self.canvas = MplCanvas(self, width=5, heigth=4, dpi=100)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        self.window.graph_Layout.addWidget(toolbar)
        self.window.graph_Layout.addWidget(self.canvas)
        self.window.show_graph_btn.clicked.connect(self.show_DataPlot)
        
        self.window.show()

    def SetupUI(self):
        ui_file_name = resource_path("client/uiform/form2.ui")
        ui_file = QFile(ui_file_name)
        if not ui_file.open(QIODevice.ReadOnly):
            print(f"Cannot open {ui_file_name}: {ui_file.errorString()}")
            sys.exit(-1)
        loader = QUiLoader()
        window =loader.load(ui_file)
        ui_file.close()
        if not window:
            print(loader.errorString())
            sys.exit(-1)
        return window
            
    @QtCore.Slot()
    def Input_path(self):
        edit_path = self.window.path_edit.text()
        print(f"Enter Path: {edit_path}")
        self.window.path_edit.setText(edit_path)
        
    @QtCore.Slot()
    def file_exeplore(self):
        # self.file_navi.show()
        self.select_data = self.file_navi.getOpenFileName(None, "Select File")[0]
        # print(f"select_data: {self.select_data}")
        self.window.path_edit.setText(self.select_data)
        self.window.edit_path_print.setText(self.select_data)

    @QtCore.Slot()
    def Select_data(self):
        # print(f"select_data: {self.select_data}")
        self.window.path_edit.setText(self.select_data)
        self.window.edit_path_print.setText(self.select_data)

        if os.path.splitext(self.select_data)[1] == '.csv':
            self.Input_data = pd.read_csv(self.select_data)
        elif os.path.splitext(self.select_data)[1] == '.xlsx':
            self.Input_data =  pd.read_excel(self.select_data)
        
        self.Input_data.reset_index()
        self.Input_data.columns = ['Index', 'date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
        self.Input_data['date'] = pd.to_datetime(self.Input_data['date'])

        # view widget table setting
        self.window.input_tableView.setColumnCount(9)
        self.window.input_tableView.setHorizontalHeaderLabels(
                        ['Index', 'date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'])
        
        #DB내부에 저장된 결과물의 갯수를 저장한다.        
        count = self.Input_data.shape[0]
        
        #갯수만큼 테이블의 Row를 생성한다.
        self.window.input_tableView.setRowCount(count)
        
        #row 리스트만큼 반복하며 Table에 DB 값을 넣는다.
        for x in range(count):
            # Input_data DataFrame 내부의 column 각 값을 변수에 저장
            Index = self.Input_data.iloc[x,0]
            date  = self.Input_data.iloc[x,1]
            HUFL  = self.Input_data.iloc[x,2].round(3)
            HULL  = self.Input_data.iloc[x,3].round(3)
            MUFL  = self.Input_data.iloc[x,4].round(3)
            MULL  = self.Input_data.iloc[x,5].round(3)
            LUFL  = self.Input_data.iloc[x,6].round(3)
            LULL  = self.Input_data.iloc[x,7].round(3)
            OT    = self.Input_data.iloc[x,8].round(3)
            
            #테이블의 각 셀에 값 입력
            self.window.input_tableView.setItem(x, 0, QTableWidgetItem(str(Index)))
            self.window.input_tableView.setItem(x, 1, QTableWidgetItem(str(date)))
            self.window.input_tableView.setItem(x, 2, QTableWidgetItem(str(HUFL)))
            self.window.input_tableView.setItem(x, 3, QTableWidgetItem(str(HULL)))
            self.window.input_tableView.setItem(x, 4, QTableWidgetItem(str(MUFL)))
            self.window.input_tableView.setItem(x, 5, QTableWidgetItem(str(MULL)))
            self.window.input_tableView.setItem(x, 6, QTableWidgetItem(str(LUFL)))
            self.window.input_tableView.setItem(x, 7, QTableWidgetItem(str(LULL)))
            self.window.input_tableView.setItem(x, 8, QTableWidgetItem(str(OT)))

    @QtCore.Slot()
    def Predict_result(self):
        header = { 
            "Content-type" : "application/json",
            "accept": "application/json",
        }
        pred_url = "http://localhost:8000/DLM/predict"

        reqData = {
                    "Index" : list(self.Input_data['Index']),
                    "date"  : list(self.Input_data['date'].astype(str)),
                    "HUFL"  : list(self.Input_data['HUFL']),
                    "HULL"  : list(self.Input_data['HULL']),
                    "MUFL"  : list(self.Input_data['MUFL']),
                    "MULL"  : list(self.Input_data['MULL']),
                    "LUFL"  : list(self.Input_data['LUFL']),
                    "LULL"  : list(self.Input_data['LULL']),
                    "OT"    : list(self.Input_data['OT']),
        }

        response = requests.post(pred_url, json=reqData, headers=header)
        print(response.status_code)
        resDict = response.json()
        col = ['date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
        data = []
        for item in col :
            data.append(resDict['prediction'][item])
        self.predict_response = pd.DataFrame(data=np.array(data).T, columns=col)

        self.predict_response = self.predict_response.reset_index()
        self.predict_response.columns = ['Index', 'date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
        self.predict_response['date'] = pd.to_datetime(self.predict_response['date'])
        self.predict_response[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']] \
            = self.predict_response[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']].astype(float)

        # view widget table setting
        self.window.predict_tableView.setColumnCount(9)
        self.window.predict_tableView.setHorizontalHeaderLabels(
                        ['Index', 'date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'])
        count = self.predict_response.shape[0]
        self.window.predict_tableView.setRowCount(count)
        for x in range(count):
            Index = self.predict_response.iloc[x,0]
            date  = self.predict_response.iloc[x,1]
            HUFL  = self.predict_response.iloc[x,2].round(3)
            HULL  = self.predict_response.iloc[x,3].round(3)
            MUFL  = self.predict_response.iloc[x,4].round(3)
            MULL  = self.predict_response.iloc[x,5].round(3)
            LUFL  = self.predict_response.iloc[x,6].round(3)
            LULL  = self.predict_response.iloc[x,7].round(3)
            OT    = self.predict_response.iloc[x,8].round(3)
            
            self.window.predict_tableView.setItem(x, 0, QTableWidgetItem(str(Index)))
            self.window.predict_tableView.setItem(x, 1, QTableWidgetItem(str(date)))
            self.window.predict_tableView.setItem(x, 2, QTableWidgetItem(str(HUFL)))
            self.window.predict_tableView.setItem(x, 3, QTableWidgetItem(str(HULL)))
            self.window.predict_tableView.setItem(x, 4, QTableWidgetItem(str(MUFL)))
            self.window.predict_tableView.setItem(x, 5, QTableWidgetItem(str(MULL)))
            self.window.predict_tableView.setItem(x, 6, QTableWidgetItem(str(LUFL)))
            self.window.predict_tableView.setItem(x, 7, QTableWidgetItem(str(LULL)))
            self.window.predict_tableView.setItem(x, 8, QTableWidgetItem(str(OT)))

    @QtCore.Slot()
    def show_DataPlot(self):
        # show data graph & update
        self.canvas.axes.cla() # clear canvers
        if (self.Input_data.shape[0] > 0) and (self.Input_data.shape[1] > 0):
            self.Input_data = self.Input_data.round(3)
            input_line = self.canvas.axes.plot(self.Input_data.iloc[:,1],self.Input_data.iloc[:,8]
                                  ,lw=1, color="blue"
                                  ,marker='o',markerfacecolor='yellow', markersize=0.5, markeredgewidth=0.1)
        if (self.predict_response.shape[0] > 0) and (self.predict_response.shape[1] > 0):
            self.predict_response =  self.predict_response.round(3)
            pred_line = self.canvas.axes.plot(self.predict_response.iloc[:,1],self.predict_response.iloc[:,8]
                                  ,lw=1, ls='-',color="red"
                                  ,marker='^',markerfacecolor='yellow', markersize=0.5, markeredgewidth=0.1)
        ymax = max(self.Input_data.iloc[:,8])*1.2
        ymin = min(self.Input_data.iloc[:,8])*1.2
        self.canvas.axes.vlines(self.Input_data.iloc[-1,1],ymax=ymax, ymin=ymin, color="yellow")
        # xticks= self.canvas.axes.get_xticks()
        # xticksSet=self.canvas.axes.set_xticks(xticks)
        # ticks_font={'fontsize':4, 'fontweight':'bold'}
        # self.canvas.axes.set_xticklabels(xticks, rotation = 70, fontdict=ticks_font)
        title_font={'fontsize':10, 'fontweight':'bold'}
        self.canvas.axes.set_title("Electric Transformer Oil Temperature Sensor Predict", fontdict=title_font)
        self.canvas.axes.grid(lw=1,ls=':', alpha=0.25)
        self.canvas.axes.yaxis.grid(True, color='black')
        self.canvas.draw()

    # def show_predict_server DB data
        # 임시 DB 연결과 테이블 조회
        # with self.engine.connect() as conn:
        #     rows = conn.execute(select(
        #                 ETT_H_1.Index,

        #                 ).limit(15)).all()
        # self.graph.show()

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def appRun():
    app = QApplication(sys.argv)
    view = Client()
    sys.exit(app.exec())
   
if __name__ == "__main__":
    appRun()

