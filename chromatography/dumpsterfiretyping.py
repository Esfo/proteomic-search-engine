import pandas as pd
import pyautogui
from pynput.mouse import Listener
import time

readfile = 'C:/Base/data/190701_F350_yM2.method.csv'

csv = pd.read_csv(readfile)

nclicks = 5
clicklocs = []
def on_click(x, y, button, pressed):
    if pressed:
        print(f'{x, y}')
        clicklocs.append([x,y])
        return False

for _ in range(nclicks+1):
    with Listener(on_click=on_click) as listener:
        print('Click')
        listener.join()


columnorder = ['duration', 'newflow', 'newpercbmild']

for ind, row in csv.iterrows():
    items = row.loc[columnorder].tolist()
    items = [str(i) for i in items]
    print(items)
    
    time.sleep(0.1)
    
    pyautogui.click(*clicklocs[0])
    time.sleep(0.1)
    pyautogui.typewrite(items[0])
    time.sleep(0.1)
    
    pyautogui.click(*clicklocs[1])
    time.sleep(0.1)
    pyautogui.typewrite(items[1])
    time.sleep(0.1)
    
    pyautogui.click(*clicklocs[2])
    time.sleep(0.1)
    pyautogui.typewrite(items[2])
    time.sleep(0.1)
    
    pyautogui.click(*clicklocs[3])
    time.sleep(0.1)
    
    pyautogui.click(*clicklocs[4])
    time.sleep(0.1)