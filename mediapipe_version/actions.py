import pyautogui

pyautogui.FAILSAFE = False

def move(x, y):
    pyautogui.moveTo(x, y, duration=0)

def left_click():
    pyautogui.click()

def right_click():
    pyautogui.click(button='right')

def scroll_up():
    pyautogui.scroll(100)

def scroll_down():
    pyautogui.scroll(-100)
