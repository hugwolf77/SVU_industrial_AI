
# client app by pyside6

import sys
import os
from PySide6 import QtCore
from PySide6.QtCore import QFile, QIODevice 
from PySide6.QtWidgets import (QApplication, QWidget, QFileDialog, QTableWidget, QTableWidgetItem)
from PySide6.QtUiTools import QUiLoader

from sqlalchemy import select
from DB.connect import conn_DB
from DB.DBmodel.dataTB import dataTB


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
        self.window.setWindowTitle('Data Viewer')
        self.window.Do.clicked.connect(self.Input_path)
        self.window.DataShow.clicked.connect(self.DataPlot)
        self.window.navi_file_path.clicked.connect(self.file_exeplore)
        
        # DB connect
        self.engine = conn_DB()
        self.setTables()
        
        
        self.window.show()
        
    def SetupUI(self):
        ui_file_name = resource_path("uirorm/form.ui")
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
        edit_path = self.window.File_Path_Edit.text()
        print(f"Enter Path: {edit_path}")
        self.window.file_path_print.setText(edit_path)
        
    @QtCore.Slot()
    def file_exeplore(self):
        # self.file_navi.show()
        navi_path = self.file_navi.getOpenFileName(None, "Select File")[0]
        print(f"Navi Path: {navi_path}")
        self.window.File_Path_Edit.setText(navi_path)
        self.window.file_path_print.setText(navi_path)
        
    
    @QtCore.Slot()
    def DataPlot(self):
        self.graph.show()

    @QtCore.Slot()
    def setTables(self):
        
        # view widget table setting
        self.window.tableWidget.setColumnCount(7)
        self.window.tableWidget.setHorizontalHeaderLabels(
                        ["Index", "TIMESTAMP", "FLOW_ID", "SOURCE_IP",
                         "SOURCE_PORT", "DESTINATION_IP", "DESTINATION_PORT"])
        
        # 임시 DB 연결과 테이블 조회
        with self.engine.connect() as conn:
            rows = conn.execute(select(
                        dataTB.Index,

                        ).limit(50)).all()

        #DB내부에 저장된 결과물의 갯수를 저장한다.        
        count = len(rows)
        
        #갯수만큼 테이블의 Row를 생성한다.
        self.window.tableWidget.setRowCount(count)
        
        #row 리스트만큼 반복하며 Table에 DB 값을 넣는다.
        for x in range(count):
            #리스트 내부의 column쌍은 튜플로 반환하므로 튜플의 각 값을 변수에 저장
            Index, TIMESTAMP, FLOW_ID, SOURCE_IP, SOURCE_PORT, DESTINATION_IP, DESTINATION_PORT = rows[x]
            
            #테이블의 각 셀에 값 입력
            self.window.tableWidget.setItem(x, 0, QTableWidgetItem(str(Index)))

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

